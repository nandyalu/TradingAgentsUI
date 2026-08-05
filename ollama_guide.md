# Ollama / RX 6600 Tuning Guide

Notes from tuning the `ollama-pool` stack (2x RX 6600, 8GB each, fronted by a
FastAPI load balancer) that TradingAgents talks to via a single
`OLLAMA_BASE_URL`. Stack lives at `~/projects/ollama-stack/ollama-pool/`.

Started from a YouTube video ("Running a 35B AI Model on 6GB VRAM, FAST —
llama.cpp Guide" by Codacus) about squeezing MoE models onto small GPUs with
raw `llama.cpp` flags. Most of its specific flags don't map onto Ollama
1:1 — this doc is what actually applied, what didn't, and the numbers behind
each decision.

## Hardware / host reality

- 4x RX 6600, gfx1032 (needs `HSA_OVERRIDE_GFX_VERSION=10.3.0`), 8.5GB VRAM
  each (physical), ~7.4-8.0GB usable.
- **Tensor-splitting a model across these cards crashes ROCm/ggml** —
  verified with two different models, two different crash signatures.
  Consumer RDNA2 doesn't reliably support the peer-to-peer access
  multi-GPU splitting needs. Each container gets exactly one GPU.
- Host has only **15GB RAM** (not 24GB+ like most of these
  guides/benchmarks assume). This is the actual bottleneck for anything
  that doesn't fit in VRAM — more on this below.

## Architecture

```
TradingAgents → OLLAMA_BASE_URL (port 11435)
                      │
                 ollama-lb (FastAPI proxy, least-busy routing)
                      │
         ┌────────────┴────────────┐
    ollama-pool (card0)      ollama-pool-b (card1)
```

Two independent single-GPU Ollama instances, round-robined by a small
FastAPI proxy (`ollama-pool/proxy/main.py`) instead of nginx, because
TradingAgents only supports one shared endpoint for both its "deep" and
"quick" thinking models — this is how a second GPU gets used at all
without any TradingAgents code changes.

## Proxy bug (fixed) — was silently truncating responses

The original proxy used `async with client.stream(...) as resp:` and
returned the `StreamingResponse` from inside that block. In Starlette,
`return` triggers `__aexit__` immediately — closing the upstream
connection *before* the response body was actually streamed to the
client, while still forwarding the upstream's original `Content-Length`
header. The resulting exception was swallowed by a bare `finally: return`
in the generator.

Symptom: intermittent `200 OK` with an **empty body**, which showed up on
the TradingAgents side as `json.decoder.JSONDecodeError: Expecting value:
line 1 column 1`.

Fix: keep the httpx response open until the streaming generator is fully
drained, release the backend concurrency slot only then (this also fixed
`CONCURRENCY_PER_BACKEND` actually throttling in-flight generations,
which it wasn't doing before — the slot was released as soon as headers
arrived, not when generation finished).

If you ever see that JSONDecodeError again, check `docker logs
ollama-proxy` for `RuntimeError: Response content shorter than
Content-Length` first — that's this bug, not a model problem.

## What the video's tricks actually map to on Ollama

| Video technique (raw llama.cpp) | Ollama equivalent | Verdict |
|---|---|---|
| `--no-mmap` | `OLLAMA_NO_MMAP=1` | This ROCm build already runs with mmap off by default for these models — no measurable effect. Removed from compose. |
| `--ctk`/`--ctv` (turbo-quant KV cache) | `OLLAMA_KV_CACHE_TYPE=q8_0`/`q4_0` | Real, but only matters if it's aggressive enough to change GPU layer offload (see below) |
| flash attention | `OLLAMA_FLASH_ATTENTION=1` | Already auto-enabled by this Ollama build regardless of the env var — harmless to set explicitly, no measured effect |
| `--n-cpu-moe` (expert-level MoE offload) | **No equivalent.** Open Ollama issue ([#11772](https://github.com/ollama/ollama/issues/11772)). Ollama only does coarse *whole-layer* offload, not per-expert. | This is the one that actually matters for real MoE models and Ollama can't do it — see qwen3.6 results below. |
| `--mlock` | No confirmed env var found | Not tested |

**Key finding: KV cache quantization only helps if it's aggressive enough
to flip a layer from CPU back to GPU.** `q8_0` bought back ~100MB —
not enough. `q4_0` bought back enough to close the gap. Check
`docker logs ollama-pool | grep offloaded` after any change — "N/M
layers to GPU" where N < M means you're leaving speed on the table, and
partial-N tests here (see qwen3.5:9b) show it's not a small penalty.

## qwen3.5:9b (dense, 9B) — the actual production model

Was stuck at **32/33 layers** on GPU (`offloading output layer to CPU`
in the logs) at 8192 context — the CPU-resident output/lm_head layer
(large vocab projection) was the bottleneck, not a minor rounding issue.

| Config | GPU layers | Solo tok/s | Concurrent tok/s (both cards) |
|---|---|---|---|
| Baseline (no env vars) | 32/33 | 18.1 | ~15.3 |
| `OLLAMA_KV_CACHE_TYPE=q8_0` | 32/33 | 18.0 | ~15.5 |
| `OLLAMA_KV_CACHE_TYPE=q4_0` | **33/33** | **29.0** | **~28** |

**~60% throughput increase**, from closing that last layer's VRAM gap —
not a rounding error. Quality-checked with `"think": false`: coherent,
complete output, no visible degradation from the 4-bit KV cache at normal
prompt lengths.

Current live config on both `ollama-pool` and `ollama-pool-b`:
```yaml
environment:
  - OLLAMA_CONTEXT_LENGTH=${OLLAMA_CONTEXT_LENGTH:-8192}
  - OLLAMA_FLASH_ATTENTION=1
  - OLLAMA_KV_CACHE_TYPE=q4_0
```

## gemma4:e2b (dense, 5.1B) — already fine, has huge context headroom

Not a MoE model despite the "e2b" naming — it's a MatFormer/PLE dense
model (elastic sub-model of Gemma 4, no routed experts). `--n-cpu-moe`
would never apply to it. Already fits fully at 36/36 layers with room to
spare: ~7.2GB used of ~8.5GB card, ~70 tok/s solo, barely drops under
concurrent load (~65-70 tok/s both cards at once).

**Context length headroom test** (via per-request `num_ctx`, not the
container-wide `OLLAMA_CONTEXT_LENGTH`):

| num_ctx | GPU layers | Total VRAM |
|---|---|---|
| 8192 (current default) | 36/36 | 7.2 GiB |
| 32768 | 36/36 | 7.3 GiB |
| 65536 | 36/36 | 7.5 GiB |
| 98304 | 36/36 | 7.7 GiB |
| **126976** | **36/36** | **7.9 GiB** |
| 131072 (model's own max) | 35/36 — falls off GPU | — |

Can go **~15x** past the current default without losing GPU offload — but
only if the *caller* passes `num_ctx` explicitly in the request. Important
gotcha: `OLLAMA_CONTEXT_LENGTH` is a per-*container* setting shared by
every model that container serves (gemma4:e2b **and** qwen3.5:9b both run
on `ollama-pool`). Raising it globally to benefit gemma4:e2b would also
raise qwen3.5:9b's default and could knock it back below 33/33 layers.
Per-request `num_ctx` avoids this entirely — each model's runner sizes
independently based on what that specific request asks for. Left the
container default alone; if gemma4:e2b needs more context somewhere,
set `num_ctx` on that specific call, not the compose env var.

## qwen3.6 (36B MoE, 23GB quantized) — the model the video was actually about

This is the real test of the video's premise: a MoE model too big for
8GB VRAM. Result was not good on this hardware:

- Ollama's scheduler only got **12/41 layers onto GPU** — and it's doing
  *whole-layer* offload (all of a layer's experts move together), not
  the expert-level split `--n-cpu-moe` does in raw llama.cpp. This is
  the gap noted above: Ollama has no equivalent lever.
- **5 tok/s** — usable but far from the video's 17-29 tok/s on similar
  hardware, because of the coarser offload.
- Loading it pushed **16.2GB of CPU-resident weights** onto a host with
  only ~8GB free RAM at the time. Swap usage spiked from 2.8GB to
  **16GB** during a single generation.

**Do not run qwen3.6 (or any model this size) on both `ollama-pool` and
`ollama-pool-b` at the same time.** Two concurrent loads would need
~32GB of CPU-resident weight RAM combined — this host has 15GB total.
That's an OOM risk for every other container on the box (trawl, stash,
gluetun, portrait-studio, dockge, etc.), not just a slowdown. Tested this
once, unloaded immediately after (`curl .../api/generate -d
'{"model":"qwen3.6:latest","keep_alive":0}'`), confirmed the other
containers survived.

If qwen3.6-class MoE models become a real requirement, the actual fix is
standing up a raw `llama.cpp`/ROCm container (bypassing Ollama) to get
`--n-cpu-moe` control — scoped but not built. On this specific host,
RAM (not VRAM, not GPU count) is the ceiling for that path too.

## Benchmarking methodology (reusable)

Get real tok/s from Ollama's own metrics rather than wall-clock timing
(which includes model load time on a cold start):

```bash
curl -s http://localhost:11435/api/generate -H "Content-Type: application/json" \
  -d '{"model":"<model>","prompt":"<prompt>","stream":false,"think":false,"options":{"num_predict":<N>}}' \
  | python3 -c "
import json, sys
d = json.load(sys.stdin)
ec, ed = d['eval_count'], d['eval_duration']
print(f'{ec} tokens in {ed/1e9:.2f}s = {ec/(ed/1e9):.2f} tok/s')
"
```

- Always **warm up first** (one throwaway request) — the first call after
  a container restart pays model-load time, which dwarfs generation time.
- `"think": false` at the top level (not nested under `options`) actually
  disables reasoning traces for hybrid-thinking models — otherwise the
  whole `num_predict` budget gets spent on the `thinking` field and
  `response` comes back empty (not a bug, just budget exhaustion).
- Check `GET /healthz` on the proxy mid-request (`active` count per
  backend) to confirm requests are actually landing on both GPUs
  concurrently, not queuing on one.
- To hit one specific backend directly (bypass the proxy's routing):
  `docker run --rm --network ollama-pool_default curlimages/curl:latest
  -s http://ollama-pool-b:11434/api/generate ...`
- Check `docker logs ollama-pool | grep -E "offloaded|kv cache|total
  memory"` after any config or model change — this is the ground truth
  for GPU layer placement, not the tok/s number alone.

## Operational note

`OLLAMA_CONTEXT_LENGTH` is set per-container in compose, defaulting to
8192 via `${OLLAMA_CONTEXT_LENGTH:-8192}`. The repo copy of this compose
file lives at `~/projects/ollama-stack/ollama-pool/compose.yaml` — dockge
may be tracking a separate copy of it. If you edit one and restart via
the other, they can drift out of sync silently. Worth checking both
before assuming a config change actually applied.

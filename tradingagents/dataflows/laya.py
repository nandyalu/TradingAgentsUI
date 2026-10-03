"""Ask a laya sidecar typed questions about a piece of text.

laya (https://github.com/NandhaKishorM/laya) is a small encoder model. It
answers yes/no, score and choice questions in one forward pass. It generates
no text. ``laya-serve`` exposes it over HTTP as ``POST /v1/systemone``.

Set ``LAYA_URL`` to the sidecar's address to turn grading on. Set
``LAYA_API_KEY`` too if the sidecar requires a bearer token. When ``LAYA_URL``
is unset, or the sidecar fails, ``ask`` returns None and the caller continues
without a grade. A grade is an annotation. Nothing may depend on it being there.
"""
from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)


def laya_url() -> str | None:
    """Return the sidecar address from ``LAYA_URL``, or None if unset."""
    return os.environ.get("LAYA_URL", "").strip().rstrip("/") or None


def ask(state: dict, questions: dict, *, model: str | None = None, timeout: float = 60) -> dict | None:
    """Send one request and return its ``answers``, or None.

    ``state`` is the text, as a dict of named fields. ``questions`` uses laya's
    schema: ``{"name": {"type": "noul", "instructions": "..."}}``. ``model``
    names a checkpoint (``english``, ``multilingual`` or ``typed-decisions``).
    Without it, laya picks one from the language of the text.

    The sidecar refuses a request with 503 when it is full. This waits and
    tries again, up to three times, because the four analysts run at once.
    """
    url = laya_url()
    if not url:
        return None
    body: dict = {"state": state, "questions": questions}
    if model:
        body["model"] = model
    headers = {"content-type": "application/json"}
    key = os.environ.get("LAYA_API_KEY", "").strip()
    if key:
        headers["authorization"] = f"Bearer {key}"
    data = json.dumps(body).encode()
    for attempt in range(4):
        request = urllib.request.Request(f"{url}/v1/systemone", data=data, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.load(response).get("answers")
        except urllib.error.HTTPError as err:
            if err.code == 503 and attempt < 3:
                time.sleep(0.5 * (attempt + 1))
                continue
            logger.warning("laya refused a request: HTTP %s", err.code)
        except (OSError, ValueError) as err:
            logger.warning("laya did not answer: %s", err)
        return None
    return None

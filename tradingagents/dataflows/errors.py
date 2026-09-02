"""Vendor data-error taxonomy.

A single hierarchy so the routing layer reacts by *behavior*, not by vendor:
every condition where a vendor cannot return usable data derives from
``VendorError``, and the router catches the base types. A new vendor raises
these (or a thin vendor-named subclass) and needs no new ``except`` clause.

    VendorError
    ├── NoMarketDataError          no usable rows (empty result OR stale data)
    ├── VendorRateLimitError       transient throttle -> skip to next vendor
    ├── VendorNotConfiguredError   missing API key/config -> vendor unavailable
    └── BadVendorArgumentError     the *caller* asked for something invalid

The number of types is the number of distinct router reactions, not the number
of human-describable causes: empty and stale data get identical handling, so
they share ``NoMarketDataError`` and differ only in the free-text ``detail``.
"""

from __future__ import annotations


class VendorError(Exception):
    """Base for any condition where a vendor could not return usable data."""


class NoMarketDataError(VendorError):
    """A vendor returned no usable rows for a symbol (empty result or stale data).

    Carries both the symbol the user requested and the canonical symbol the
    vendor was actually queried with, plus a free-text ``detail``, so callers
    can build a clear message instead of emitting a vendor-specific empty
    string into the data channel.
    """

    def __init__(self, symbol: str, canonical: str | None = None, detail: str = ""):
        self.symbol = symbol
        self.canonical = canonical or symbol
        self.detail = detail
        msg = f"No market data for {symbol!r}"
        if canonical and canonical != symbol:
            msg += f" (queried as {canonical!r})"
        if detail:
            msg += f": {detail}"
        super().__init__(msg)


class VendorRateLimitError(VendorError):
    """A vendor throttled the request; the router skips to the next vendor."""


class VendorNotConfiguredError(VendorError, ValueError):
    """A vendor was selected but its API key/configuration is missing.

    Also a ``ValueError`` so existing callers that catch ``ValueError`` keep
    working while the routing layer can treat it as "vendor unavailable".
    """


class BadVendorArgumentError(VendorError, ValueError):
    """The caller asked for something that does not exist.

    **The vendor is healthy. The request was wrong.** That distinction is the
    whole reason this type exists, and it changes three things in the router:

    1. **It does not trip the circuit breaker.** The breaker exists to skip a
       vendor that is *down*, and its own docstring says only transient errors
       should open it. Counting a bad argument as vendor ill-health poisons a
       working vendor for every later call.
    2. **It does not fall through to the next vendor.** Every vendor will
       reject the same invalid argument, so trying them in turn only wastes
       requests and buries the message that explains the problem.
    3. **It reaches the model.** These messages name the valid values, so an
       agent shown one can retry correctly. Swallowing it into "no vendor
       available" throws away the answer.

    This was not hypothetical. On 2026-09-02 a model asked for an indicator
    called ``macd_histogram`` — the real name is ``macdh``. It made five such
    requests, tripped yfinance's breaker at the third, and every subsequent
    indicator call for the next five minutes failed with "No available vendor",
    **including the valid ones**. Two complete forty-minute analyses were lost
    to a typo whose correction was in the first error message.

    ``valid`` carries the accepted values where the vendor knows them, so a
    caller can render them without parsing the message.
    """

    def __init__(self, message: str, valid: list[str] | None = None):
        self.valid = list(valid or [])
        super().__init__(message)

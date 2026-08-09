"""Network calls that degrade gracefully instead of breaking the book.

This module is the reason the Data Management chapters can be built, read and re-run
without a working connection to PubChem or NIST.

Why it exists
-------------
These chapters teach data access by actually accessing data, which means the book's build
depends on public services staying up and staying quick. They do not: PubChem in
particular rate-limits, and repeated builds have produced 502s and read timeouts. Because
`_config.yml` sets `allow_errors: false`, a single such failure fails the entire book, and
`make publish` refuses to deploy.

So every live call in these chapters goes through here. Each one tries the network first;
if the call raises, or the server returns a 5xx, the stored copy captured on a previous
successful run is returned instead, with a visible note saying so.

What is and is not treated as failure
-------------------------------------
A 4xx is a real answer from the server — "no such compound" is exactly what Topic 4.2 uses
to teach error handling — so 4xx responses are passed through untouched. Only exceptions
and 5xx responses trigger the fallback, because those mean the service, not the request,
is the problem.

Keeping the cache fresh
-----------------------
A response is stored the first time a call succeeds, and then left alone. It is deliberately
*not* refreshed on every successful build: the pages here carry rotating Cloudflare tokens
that change between requests, so re-storing them would dirty the working tree on every
build without changing a single value anyone reads.

The cache is a fallback, not a mirror — it only matters on the days the service is
unreachable. To refresh it after an API genuinely changes, delete the file (or the whole
`data/api_cache/` directory) and run the chapter once while online.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import re
from types import SimpleNamespace

import requests

CACHE_DIR = pathlib.Path(__file__).parent / "data" / "api_cache"

__all__ = ["safe_get", "cached_json", "cached_record", "CACHE_DIR"]


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------

def _key(url: str, params: dict | None) -> str:
    """A readable, stable filename for a request."""
    tail = re.sub(r"[^A-Za-z0-9]+", "_", url.split("://", 1)[-1])[-60:].strip("_")
    digest = hashlib.sha1(
        (url + json.dumps(params or {}, sort_keys=True)).encode()
    ).hexdigest()[:8]
    return f"{tail}__{digest}"


def _store(path: pathlib.Path, payload: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")


class _CachedResponse:
    """The parts of `requests.Response` these chapters actually use."""

    def __init__(self, text: str, status_code: int, url: str):
        self.text = text
        self.status_code = status_code
        self.url = url
        self.from_cache = True

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 300

    def json(self):
        return json.loads(self.text)

    def raise_for_status(self):
        if not self.ok:
            raise requests.HTTPError(f"{self.status_code} for {self.url}", response=None)


# ---------------------------------------------------------------------------
# public helpers
# ---------------------------------------------------------------------------

def safe_get(url, params=None, timeout=20, **kwargs):
    """`requests.get` that falls back to a stored copy when the service misbehaves.

    Returns something that quacks like a `requests.Response`: `.text`, `.status_code`,
    `.ok`, `.json()` and `.raise_for_status()` all work either way.
    """
    path = CACHE_DIR / f"{_key(url, params)}.json"

    try:
        response = requests.get(url, params=params, timeout=timeout, **kwargs)
        if response.status_code < 500:
            # A 4xx is a genuine answer worth caching and worth showing the reader.
            if not path.exists():
                _store(path, {"url": url, "status_code": response.status_code,
                              "text": response.text})
            return response
        reason = f"HTTP {response.status_code}"
    except Exception as exc:                      # noqa: BLE001 — any network trouble
        reason = type(exc).__name__

    if path.exists():
        blob = json.loads(path.read_text(encoding="utf-8"))
        print(f"[cached] {reason} from {url.split('/')[2]} — using the stored response")
        return _CachedResponse(blob["text"], blob["status_code"], url)

    raise RuntimeError(
        f"{url} failed ({reason}) and no cached copy exists at {path}. "
        "Run this cell once while online to populate the cache."
    )


def cached_json(key: str, producer):
    """Cache the JSON-serializable result of `producer()`.

    For calls that are not plain HTTP GETs — a library wrapper, or a multi-step routine
    whose *result* is what matters rather than the raw response.
    """
    path = CACHE_DIR / f"{key}.json"
    try:
        value = producer()
        if not path.exists():
            _store(path, {"key": key, "value": value})
        return value
    except Exception as exc:                      # noqa: BLE001
        if path.exists():
            print(f"[cached] {key}: {type(exc).__name__} — using the stored result")
            return json.loads(path.read_text(encoding="utf-8"))["value"]
        raise


def cached_record(key: str, producer, fields):
    """Like `cached_json`, but for an object: keeps `fields` and returns a namespace.

    Library objects (a `pubchempy.Compound`, say) cannot be stored directly, but the few
    attributes a chapter reads from them can. Attribute access on the result works the
    same whether it came from the network or from disk.
    """
    def _pull():
        obj = producer()
        return {f: getattr(obj, f) for f in fields}

    return SimpleNamespace(**cached_json(key, _pull))

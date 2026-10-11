"""Egress guard for proof runs: refuse an outgoing request BEFORE it is sent.

Off unless EGRESS_GUARD_POLICY is set (app/main.py installs it at startup). Used for the
owner-approved proof runs against paid and external services, so that what the app may
send is fixed in advance: approved hosts only, a request cap per host and in total, and
a spend cap. A request that would break any of them is refused before a byte leaves the
process, and from then on every external request is refused (a hard stop).

Hook points (each request attempt passes through one of them, so retries and redirects
count as requests):
- httpx2: `httpx2.HTTPTransport.handle_request` and `AsyncHTTPTransport.handle_async_request`.
  httpx2 (pydantic/httpx2, with its core httpcore2) is a separately published successor
  of httpx; the OpenAI SDK 3.x sends everything through it.
- httpx: `httpx.HTTPTransport.handle_request` and the async one. Used by Langfuse's REST
  client and by qdrant-client (internal host, not counted).
- urllib3: `urllib3.connectionpool.HTTPConnectionPool.urlopen` (HTTPS pools subclass it).
  The OpenTelemetry OTLP exporter that ships Langfuse traces uses urllib3; so does
  `requests`. Every urlopen call is one attempt (urllib3's own retries call it again).
- DNS: `socket.getaddrinfo`, as a second layer: only approved and internal host names
  resolve, so a client this module does not know about cannot reach anything else.

Spend cap: before an OpenAI request is sent, its worst-case cost is added to a running
bound: every byte of the request body counted as one input token (a token is never
shorter than one byte, so this over-counts) and the request's `max_completion_tokens` as
output (16,384 if absent). If the bound would pass `spend_cap_usd`, the request is
refused. The real spend is in llm_usage afterwards; this bound is only ever higher.

Policy (JSON in EGRESS_GUARD_POLICY):
  {"hosts": {"api.openai.com": 24, "$LANGFUSE_HOST": 8},   # per-host request caps
   "total": 32,                                              # cap over all external hosts
   "internal_hosts": ["postgres", "qdrant", "redis", "localhost", "127.0.0.1", "::1"],
   "spend_cap_usd": 0.10, "paid_host": "api.openai.com",
   "prices_per_1m": {"chat_input": 0.15, "chat_output": 0.60, "embedding": 0.02}}
A host written as "$NAME" is read from that environment variable (its host part), so a
policy never has to spell out a value from .env. Every decision is logged to stderr as
one `egress-guard:` line (host, path, counts, spend bound); bodies and headers never are.
"""

import json
import os
import socket
import sys
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any
from urllib.parse import urlsplit

DEFAULT_MAX_OUTPUT_TOKENS = 16_384  # gpt-4o-mini's output limit, used when a body sets none


class EgressRefused(RuntimeError):
    """The guard refused a request before it was sent."""


@dataclass
class Policy:
    hosts: dict[str, int]
    total: int
    internal_hosts: frozenset[str]
    spend_cap_usd: Decimal
    paid_host: str = "api.openai.com"
    prices_per_1m: dict[str, Decimal] = field(default_factory=dict)

    @classmethod
    def from_json(cls, text: str, env: Mapping[str, str] | None = None) -> "Policy":
        env = os.environ if env is None else env
        raw = json.loads(text)
        hosts: dict[str, int] = {}
        for name, cap in raw["hosts"].items():
            if name.startswith("$"):
                value = env.get(name[1:], "")
                if not value:
                    raise ValueError(f"policy host {name}: that environment variable is empty")
                name = urlsplit(value).hostname or value
            hosts[name.lower()] = int(cap)
        prices = {k: Decimal(str(v)) for k, v in raw.get("prices_per_1m", {}).items()}
        return cls(
            hosts=hosts,
            total=int(raw["total"]),
            internal_hosts=frozenset(h.lower() for h in raw.get("internal_hosts", [])),
            spend_cap_usd=Decimal(str(raw["spend_cap_usd"])),
            paid_host=raw.get("paid_host", "api.openai.com").lower(),
            prices_per_1m=prices,
        )


@dataclass
class Guard:
    policy: Policy
    log: Callable[[str], None] = lambda line: print(line, file=sys.stderr, flush=True)
    counts: dict[str, int] = field(default_factory=dict)
    total: int = 0
    internal: int = 0
    refused: int = 0
    spend_bound_usd: Decimal = Decimal("0")
    tripped: str | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def check(self, host: str | None, method: str, path: str, body: bytes | None) -> None:
        """Count one request attempt, or raise EgressRefused BEFORE it is sent."""
        name = (host or "").lower().strip("[]")
        with self._lock:
            if name in self.policy.internal_hosts:
                self.internal += 1
                return
            reason = self._refusal(name, path, body)
            if reason is not None:
                self.refused += 1
                if self.tripped is None:
                    self.tripped = reason
                self.log(f"egress-guard: REFUSED {method} {name}{path}: {reason}")
                raise EgressRefused(f"egress guard refused {method} {name}{path}: {reason}")
            self.counts[name] = self.counts.get(name, 0) + 1
            self.total += 1
            self.spend_bound_usd += self._cost_bound(name, path, body)
            self.log(
                f"egress-guard: ALLOW #{self.total} {method} {name}{path} "
                f"(host {self.counts[name]}/{self.policy.hosts[name]}, "
                f"total {self.total}/{self.policy.total}, "
                f"spend bound ${self.spend_bound_usd:.6f}/${self.policy.spend_cap_usd})"
            )

    def _refusal(self, name: str, path: str, body: bytes | None) -> str | None:
        if self.tripped is not None:
            return f"hard stop after an earlier refusal ({self.tripped})"
        if name not in self.policy.hosts:
            return "host not approved"
        if self.counts.get(name, 0) + 1 > self.policy.hosts[name]:
            return f"host cap {self.policy.hosts[name]} reached"
        if self.total + 1 > self.policy.total:
            return f"total cap {self.policy.total} reached"
        bound = self.spend_bound_usd + self._cost_bound(name, path, body)
        if bound > self.policy.spend_cap_usd:
            return f"spend bound ${bound:.6f} would pass the cap ${self.policy.spend_cap_usd}"
        return None

    def _cost_bound(self, name: str, path: str, body: bytes | None) -> Decimal:
        """Worst-case USD of one request to the paid host (0 for any other host)."""
        if name != self.policy.paid_host:
            return Decimal("0")
        size = len(body or b"")
        price = self.policy.prices_per_1m
        if path.endswith("/embeddings"):
            return Decimal(size) * price.get("embedding", Decimal("0")) / 1_000_000
        max_out = DEFAULT_MAX_OUTPUT_TOKENS
        try:
            parsed = json.loads(body or b"{}")
            value = parsed.get("max_completion_tokens") or parsed.get("max_tokens")
            if isinstance(value, int) and value > 0:
                max_out = value
        except (ValueError, AttributeError):
            pass
        return (
            Decimal(size) * price.get("chat_input", Decimal("0"))
            + Decimal(max_out) * price.get("chat_output", Decimal("0"))
        ) / 1_000_000

    def allows_dns(self, host: str | None) -> bool:
        name = (host or "").lower().strip("[]")
        return not name or name in self.policy.hosts or name in self.policy.internal_hosts

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "counts": dict(self.counts),
                "total": self.total,
                "internal": self.internal,
                "refused": self.refused,
                "spend_bound_usd": str(self.spend_bound_usd),
                "tripped": self.tripped,
            }


# --- installing the hooks -------------------------------------------------------------------------

_installed: Guard | None = None
_originals: list[tuple[Any, str, Any]] = []


def _httpx_body(request: Any) -> bytes | None:
    try:
        content = request.content  # raises for an unread streaming body
        return bytes(content) if content else None
    except Exception:
        return None


def _patch(owner: Any, name: str, wrapper: Callable[[Any], Any]) -> None:
    original = getattr(owner, name)
    _originals.append((owner, name, original))
    setattr(owner, name, wrapper(original))


def install(guard: Guard) -> Guard:
    """Hook httpx2, httpx, urllib3 and DNS. Idempotent per process (a second call raises)."""
    global _installed
    if _installed is not None:
        raise RuntimeError("the egress guard is already installed")

    def sync_hook(original: Any) -> Any:
        def handle_request(self: Any, request: Any) -> Any:
            guard.check(request.url.host, request.method, request.url.path, _httpx_body(request))
            return original(self, request)

        return handle_request

    def async_hook(original: Any) -> Any:
        async def handle_async_request(self: Any, request: Any) -> Any:
            guard.check(request.url.host, request.method, request.url.path, _httpx_body(request))
            return await original(self, request)

        return handle_async_request

    import httpx
    import httpx2
    import urllib3.connectionpool

    for module in (httpx2, httpx):
        _patch(module.HTTPTransport, "handle_request", sync_hook)
        _patch(module.AsyncHTTPTransport, "handle_async_request", async_hook)

    def urlopen_hook(original: Any) -> Any:
        def urlopen(
            self: Any, method: str, url: str, body: Any = None, *args: Any, **kw: Any
        ) -> Any:
            data = body if isinstance(body, bytes) else None
            guard.check(self.host, method, urlsplit(url).path or url, data)
            return original(self, method, url, body, *args, **kw)

        return urlopen

    _patch(urllib3.connectionpool.HTTPConnectionPool, "urlopen", urlopen_hook)

    def dns_hook(original: Any) -> Any:
        def getaddrinfo(host: Any, *args: Any, **kw: Any) -> Any:
            name = host.decode() if isinstance(host, bytes) else host
            if not guard.allows_dns(name):
                guard.log(f"egress-guard: REFUSED DNS lookup of {name!r}: host not approved")
                raise EgressRefused(f"egress guard refused a DNS lookup of {name!r}")
            return original(host, *args, **kw)

        return getaddrinfo

    _patch(socket, "getaddrinfo", dns_hook)
    _installed = guard
    guard.log(
        f"egress-guard: installed; caps {guard.policy.hosts}, total {guard.policy.total}, "
        f"spend cap ${guard.policy.spend_cap_usd}"
    )
    return guard


def uninstall() -> None:
    """Restore every hooked function (tests)."""
    global _installed
    while _originals:
        owner, name, original = _originals.pop()
        setattr(owner, name, original)
    _installed = None


def install_from_env() -> Guard | None:
    """Install from EGRESS_GUARD_POLICY if it is set (app startup)."""
    text = os.environ.get("EGRESS_GUARD_POLICY")
    if not text:
        return None
    return install(Guard(Policy.from_json(text)))

"""The egress guard refuses requests BEFORE they are sent. No network: the real client
libraries run, and only their last step (the connection pool that would open a socket)
is replaced by a fake that records what would have been sent.

Also measures, with the real OpenAI SDK and the real Langfuse OTLP exporter, how many
requests each kind of chat turn makes per host (the numbers the proof run's caps use).
"""

import io
import json
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import httpcore
import httpcore2
import httpx
import httpx2
import openai
import pytest
import urllib3
import urllib3.connectionpool
from urllib3.response import HTTPResponse

from app import egress_guard as eg
from app.config import get_settings
from app.ingest.embed import DenseEmbedder
from app.llm.client import ChatClient
from app.llm.usage import chat_cost
from app.rag.pipeline import ChatDeps, ChatEvent, run_chat_turn
from tests.unit.test_pipeline import (
    ABSTRACT,
    CONVERSATION,
    HISTORY,
    SECTION,
    FakeSearch,
    FakeSparse,
    FakeStore,
)

LANGFUSE = "langfuse.invalid"
OPENAI = "api.openai.com"


def _policy(**over: Any) -> eg.Policy:
    raw = {
        "hosts": {OPENAI: 3, "$LANGFUSE_HOST": 2},
        "total": 10,
        "internal_hosts": ["qdrant", "localhost", "127.0.0.1"],
        "spend_cap_usd": 1.0,
        "prices_per_1m": {"chat_input": 0.15, "chat_output": 0.60, "embedding": 0.02},
    } | over
    return eg.Policy.from_json(json.dumps(raw), env={"LANGFUSE_HOST": f"https://{LANGFUSE}"})


@pytest.fixture
def lines() -> list[str]:
    return []


@pytest.fixture
def guard(lines: list[str]) -> Iterator[eg.Guard]:
    g = eg.install(eg.Guard(_policy(), log=lines.append))
    yield g
    eg.uninstall()


# --- fake "networks": what would have been sent ------------------------------------------------


@dataclass
class FakeOpenAIServer:
    """Stands in for httpcore2's connection pool: answers like the OpenAI API would."""

    status: int = 200
    sent: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    answer: str = "Paper A adds adapters [1]. Its abstract says less [2]."

    def handle_request(self, request: httpcore2.Request) -> httpcore2.Response:
        path = request.url.target.decode()
        body = json.loads(b"".join(request.stream) or b"{}")  # type: ignore[arg-type]
        self.sent.append((path, body))
        if self.status != 200:
            return self._json(self.status, {"error": {"message": "invented", "type": "server"}})
        if path.endswith("/embeddings"):
            data = [
                {"object": "embedding", "index": i, "embedding": [0.0] * 1536}
                for i, _ in enumerate(body["input"])
            ]
            return self._json(
                200,
                {
                    "object": "list",
                    "data": data,
                    "model": body["model"],
                    "usage": {"prompt_tokens": 12, "total_tokens": 12},
                },
            )
        if body.get("stream"):
            return self._stream(self.answer)
        schema = (body.get("response_format") or {}).get("json_schema", {}).get("name")
        if schema == "PlanOutput":
            content = json.dumps(
                {
                    "intent": "qa",
                    "rephrasings": ["low-rank adapters", "LoRA"],
                    "arxiv_ids": [],
                    "categories": [],
                    "published_from": None,
                    "published_to": None,
                }
            )
        elif schema == "VerifyOutput":
            asked = re.findall(r'<sentence index="(\d+)"', body["messages"][1]["content"])
            content = json.dumps(
                {"verdicts": [{"index": int(i), "supported": True, "reason": ""} for i in asked]}
            )
        else:
            content = "What are the limitations of LoRA?"
        return self._json(
            200,
            {
                "id": "c",
                "object": "chat.completion",
                "created": 0,
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": content},
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
            },
        )

    @staticmethod
    def _json(status: int, payload: Any) -> httpcore2.Response:
        return httpcore2.Response(
            status,
            headers=[(b"content-type", b"application/json")],
            content=json.dumps(payload).encode(),
        )

    @staticmethod
    def _stream(text: str) -> httpcore2.Response:
        chunk = {
            "id": "c",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "m",
            "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
        }
        usage = {
            "id": "c",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "m",
            "choices": [],
            "usage": {"prompt_tokens": 900, "completion_tokens": 20, "total_tokens": 920},
        }
        body = f"data: {json.dumps(chunk)}\n\ndata: {json.dumps(usage)}\n\ndata: [DONE]\n\n"
        return httpcore2.Response(
            200, headers=[(b"content-type", b"text/event-stream")], content=body.encode()
        )


@pytest.fixture
def openai_server(monkeypatch: pytest.MonkeyPatch) -> FakeOpenAIServer:
    server = FakeOpenAIServer()
    monkeypatch.setattr(
        httpcore2.ConnectionPool,
        "handle_request",
        lambda self, request: server.handle_request(request),
    )
    return server


@pytest.fixture
def urllib3_sent(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str, str]]:
    """Stands in for the step where urllib3 would write the request to a socket."""
    sent: list[tuple[str, str, str]] = []

    def make_request(self: Any, conn: Any, method: str, url: str, *a: Any, **kw: Any) -> Any:
        sent.append((self.host, method, url))
        return HTTPResponse(
            body=io.BytesIO(b""),
            status=200,
            headers={},
            preload_content=False,
            request_method=method,
            request_url=url,
        )

    monkeypatch.setattr(urllib3.connectionpool.HTTPConnectionPool, "_make_request", make_request)
    return sent


def _openai_client() -> openai.OpenAI:
    # An invented key; the fake pool answers, nothing leaves the process.
    return openai.OpenAI(api_key="sk-invented-for-tests", max_retries=0)


def _chat_client() -> tuple[ChatClient, list[tuple[str, str, int, int]]]:
    rows: list[tuple[str, str, int, int]] = []

    def record(feature: str, model: str, tokens_in: int, tokens_out: int) -> Any:
        rows.append((feature, model, tokens_in, tokens_out))
        return chat_cost(tokens_in, tokens_out)

    return ChatClient(_openai_client(), record, sleep=lambda s: None), rows


MESSAGES = [{"role": "user", "content": "invented question"}]


# --- httpx2 (the OpenAI SDK) ---------------------------------------------------------------------


def test_httpx2_retries_count_and_the_request_over_the_cap_is_never_sent(
    guard: eg.Guard, openai_server: FakeOpenAIServer, lines: list[str]
) -> None:
    """Every attempt of our retry loop is a request: with cap 3, attempts 1-3 reach the
    server (503 each, so they are retried) and attempt 4 is refused before it is sent.
    The OpenAI SDK passes EgressRefused through unwrapped, so it is not retried: the
    call stops at once instead of using its 5th attempt."""
    openai_server.status = 503
    client, rows = _chat_client()
    with pytest.raises(eg.EgressRefused, match="host cap 3 reached"):
        client.complete(MESSAGES, feature="chat-condense", max_tokens=50)
    assert len(openai_server.sent) == 3
    assert guard.counts == {OPENAI: 3} and guard.refused == 1
    assert rows == []  # nothing succeeded, nothing billed
    assert any("REFUSED POST api.openai.com/v1/chat/completions: host cap 3" in x for x in lines)


def test_httpx2_unapproved_host_is_refused_before_sending(
    guard: eg.Guard, openai_server: FakeOpenAIServer
) -> None:
    with httpx2.Client() as c, pytest.raises(eg.EgressRefused, match="host not approved"):
        c.get("https://evil.example/collect")
    assert openai_server.sent == [] and guard.total == 0


def test_the_spend_bound_refuses_before_sending(
    lines: list[str], openai_server: FakeOpenAIServer
) -> None:
    # Bound for this call: body bytes as input tokens + 1000 output tokens = over $0.0005.
    eg.install(eg.Guard(_policy(spend_cap_usd=0.0005), log=lines.append))
    try:
        client, _ = _chat_client()
        with pytest.raises(eg.EgressRefused, match="spend bound"):
            client.complete(MESSAGES, feature="chat-condense", max_tokens=1000)
    finally:
        eg.uninstall()
    assert openai_server.sent == []
    assert any("would pass the cap $0.0005" in x for x in lines)


def test_after_a_refusal_everything_external_stops(
    guard: eg.Guard, openai_server: FakeOpenAIServer
) -> None:
    with httpx2.Client() as c:
        with pytest.raises(eg.EgressRefused):
            c.get("https://evil.example/")
        with pytest.raises(eg.EgressRefused, match="hard stop"):
            c.post("https://api.openai.com/v1/embeddings", json={"input": ["x"], "model": "m"})
    assert openai_server.sent == []


# --- httpx (Langfuse REST client, qdrant-client) --------------------------------------------------


def test_httpx_refuses_unapproved_hosts_and_passes_internal_ones(
    guard: eg.Guard, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent: list[str] = []

    def handle(self: Any, request: httpcore.Request) -> httpcore.Response:
        sent.append(request.url.host.decode())
        return httpcore.Response(200, content=b"{}")

    monkeypatch.setattr(httpcore.ConnectionPool, "handle_request", handle)
    with httpx.Client() as c:
        with pytest.raises(eg.EgressRefused, match="host not approved"):
            c.get("https://evil.example/")
        assert c.get("http://qdrant:6333/collections").status_code == 200  # internal
    assert sent == ["qdrant"] and guard.internal == 1 and guard.total == 0


# --- urllib3 (the Langfuse OTLP exporter) ---------------------------------------------------------


def test_urllib3_caps_and_unapproved_hosts_are_enforced_before_sending(
    guard: eg.Guard, urllib3_sent: list[tuple[str, str, str]]
) -> None:
    pool = urllib3.PoolManager()
    url = f"https://{LANGFUSE}/api/public/otel/v1/traces"
    for _ in range(2):  # the Langfuse cap in this policy is 2
        assert pool.request("POST", url, body=b"spans").status == 200
    with pytest.raises(eg.EgressRefused, match="host cap 2"):
        pool.request("POST", url, body=b"spans")
    assert len(urllib3_sent) == 2  # the third was never handed to the socket layer
    assert guard.counts == {LANGFUSE: 2}


def test_urllib3_unapproved_host_is_refused(
    guard: eg.Guard, urllib3_sent: list[tuple[str, str, str]]
) -> None:
    with pytest.raises(eg.EgressRefused, match="host not approved"):
        urllib3.PoolManager().request("GET", "https://evil.example/")
    assert urllib3_sent == []


def test_urllib3_internal_retries_each_count(
    guard: eg.Guard, monkeypatch: pytest.MonkeyPatch
) -> None:
    """urllib3's own retries call urlopen again: each attempt is counted. (GET: urllib3
    never retries a POST after a read error, since it may have been received.)"""
    attempts: list[str] = []

    def failing(self: Any, conn: Any, method: str, url: str, *a: Any, **kw: Any) -> Any:
        attempts.append(url)
        raise urllib3.exceptions.ProtocolError("invented connection reset")

    monkeypatch.setattr(urllib3.connectionpool.HTTPConnectionPool, "_make_request", failing)
    with pytest.raises(eg.EgressRefused, match="host cap 2"):
        urllib3.PoolManager(retries=urllib3.Retry(total=5)).request(
            "GET", f"https://{LANGFUSE}/api/public/health"
        )
    assert len(attempts) == 2 and guard.counts == {LANGFUSE: 2} and guard.refused == 1


# --- DNS, the second layer ---------------------------------------------------------------


def test_dns_lookups_of_unapproved_hosts_are_refused(guard: eg.Guard) -> None:
    import socket

    with pytest.raises(eg.EgressRefused):
        socket.getaddrinfo("evil.example", 443)
    assert socket.getaddrinfo("localhost", 80)  # internal names still resolve


def test_policy_reads_env_hosts_without_spelling_them_out() -> None:
    policy = _policy()
    assert policy.hosts == {OPENAI: 3, LANGFUSE: 2}
    with pytest.raises(ValueError, match="LANGFUSE_HOST"):
        eg.Policy.from_json(
            json.dumps({"hosts": {"$LANGFUSE_HOST": 1}, "total": 1, "spend_cap_usd": 0}), env={}
        )


def test_install_from_env_is_off_without_a_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EGRESS_GUARD_POLICY", raising=False)
    assert eg.install_from_env() is None


# --- measured: requests per host for each kind of chat turn -------------------------------


@pytest.fixture
def langfuse_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-measure-invented")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-measure-invented")
    monkeypatch.setenv("LANGFUSE_HOST", f"https://{LANGFUSE}")
    get_settings.cache_clear()


def test_requests_per_turn_through_the_real_sdks(
    langfuse_env: None,
    lines: list[str],
    openai_server: FakeOpenAIServer,
    urllib3_sent: list[tuple[str, str, str]],
) -> None:
    """Three turns as in the proof run, plus the abstract-only variant: the real OpenAI SDK
    (fake server) and the real Langfuse SDK + OTLP exporter (fake socket layer)."""
    from langfuse import propagate_attributes

    from app.observability.langfuse_tracer import LangfuseTracer, make_langfuse_client

    guard = eg.install(
        eg.Guard(_policy(hosts={OPENAI: 100, "$LANGFUSE_HOST": 100}, total=200), log=lines.append)
    )
    langfuse = make_langfuse_client(get_settings())
    try:
        chat, _ = _chat_client()
        openai_client = chat.client
        tracer = LangfuseTracer(langfuse, propagate_attributes)

        def turn(hits: list[Any], history: list[Any]) -> tuple[list[ChatEvent], dict[str, int]]:
            before = dict(guard.counts)
            deps = ChatDeps(
                chat=chat,
                store=FakeStore(),
                tracer=tracer,
                search_fn=FakeSearch(hits),
                dense=lambda rec: DenseEmbedder(openai_client, rec, sleep=lambda s: None),
                sparse=FakeSparse,
                record_embedding=lambda f, m, t: None,
            )
            events = list(
                run_chat_turn(
                    "What does LoRA change?",
                    conversation_id=CONVERSATION,
                    new_conversation=not history,
                    history=history,
                    deps=deps,
                    settings=get_settings(),
                )
            )
            return events, {h: guard.counts.get(h, 0) - before.get(h, 0) for h in guard.counts}

        a, count_a = turn([SECTION, ABSTRACT], [])  # (a) a normal first question
        b, count_b = turn([SECTION, ABSTRACT], HISTORY)  # (b) a follow-up: condense runs
        c, count_c = turn([], [])  # (c) a refusal: no chunks after the filters
        d, count_d = turn([ABSTRACT], [])  # (c') abstract-only sources
    finally:
        langfuse.shutdown()
        eg.uninstall()

    assert [e.event for e in a][-1] == "done" and [e.event for e in b][-1] == "done"
    assert [e.event for e in c] == ["meta", "refusal", "done"]
    assert d[-1].data["abstract_only"] is True
    # OpenAI per turn: plan 1 + embeddings 3 (question + 2 rephrasings) + generate 1 +
    # verify 1 = 6; a follow-up adds condense; a no-source refusal stops after search.
    assert (count_a[OPENAI], count_b[OPENAI], count_c[OPENAI], count_d[OPENAI]) == (6, 7, 4, 6)
    # Langfuse: one OTLP export request per turn (the flush at the trace's end).
    assert (count_a[LANGFUSE], count_b[LANGFUSE], count_c[LANGFUSE], count_d[LANGFUSE]) == (
        1,
        1,
        1,
        1,
    )
    assert all(
        host == LANGFUSE and path.endswith("/api/public/otel/v1/traces")
        for host, _, path in urllib3_sent
    )
    kinds = [p.rsplit("/", 1)[-1] for p, _ in openai_server.sent]
    assert kinds.count("embeddings") == 12 and kinds.count("completions") == 11

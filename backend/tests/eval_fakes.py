"""Fakes for the ablation runner (eval/runner.py): no network, no stores, no model.

- `auto_chat()`: a ChatClient whose fake OpenAI answers every plan, generate and verify
  request of a chat turn, however many turns run (the planner's two rephrasings included).
- `FakeJudge`: a Judge with fixed verdicts; `recall` sets the share of reference
  sentences it finds stated, so two configurations can be given different quality.
- `FakeOpenAIEmbeddings`: vectors of the dimension the CURRENT settings ask for, and a
  record of the (model, dimension, `dimensions` parameter) of every request.
- `MemoryStore`: a RunStore keeping runs and results in lists.
- `KeyedCrossEncoders`: a cross-encoder per reranker key of the current settings, like
  app/rag/rerank.get_cross_encoder, with a record of which key was loaded.
All texts and ids are invented.
"""

import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from app.config import get_settings
from app.llm.client import ChatClient, ChatUsage
from app.rag.citations import Source
from app.rag.plan import PlanOutput
from app.rag.rerank import RerankerKey, reranker_key
from app.rag.search import SearchHit
from app.rag.verify import Verdict as VerifyVerdict
from app.rag.verify import VerifyOutput
from eval.golden import GoldenItem, SetQuestion
from eval.judge import CitationItem, Judge, Verdict
from tests.chat_fakes import parsed_reply, stream_reply


def hit(name: str, arxiv_id: str = "2601.00001", chunk_type: str = "section",
        text: str | None = None) -> SearchHit:  # fmt: skip
    return SearchHit(
        chunk_id=uuid.uuid5(uuid.NAMESPACE_URL, name), arxiv_id=arxiv_id, title=f"Paper {arxiv_id}",
        section_path=None if chunk_type == "abstract" else "3 Method", chunk_type=chunk_type,
        published_at="2026-09-01T00:00:00+00:00",
        text=text or f"Invented passage {name} about low-rank adapters for fine-tuning models.",
        score=1.0,
    )  # fmt: skip


SEC = hit("sec")
ABS = hit("abs", "2601.00002", "abstract")
REFERENCE_TEXTS = {SEC.chunk_id: SEC.text}


def question(qid: str, split: Any = "generated", *, refs: bool = True,
             text: str | None = None, qtype: Any = None) -> SetQuestion:  # fmt: skip
    reference = "LoRA trains two small matrices. The weights stay frozen."
    item = GoldenItem(
        id=qid, question=text or f"Invented question {qid}?",
        reference_answer=reference if refs else None,
        reference_chunk_ids=[SEC.chunk_id] if refs else [],
        type=qtype if qtype else (None if refs else "unanswerable"),
    )  # fmt: skip
    return SetQuestion(item, split)


ANSWER = ["LoRA trains two matrices [1]. ", "The weights stay frozen [2]."]


@dataclass
class AutoCompletions:
    """Answers each request by what it asks for (a plan, a verdict list or a stream)."""

    calls: list[dict[str, Any]] = field(default_factory=list)

    def parse(self, **kwargs: Any) -> Any:
        self.calls.append({"kind": "parse", **kwargs})
        schema = kwargs["response_format"]
        if schema is PlanOutput:
            return parsed_reply(PlanOutput.model_validate(
                {"intent": "qa", "rephrasings": ["first rephrasing", "second rephrasing"],
                 "arxiv_ids": [], "categories": [], "published_from": None, "published_to": None}
            ), 300, 40)  # fmt: skip
        if schema is VerifyOutput:
            return parsed_reply(VerifyOutput(verdicts=[
                VerifyVerdict(index=0, supported=True, reason=""),
                VerifyVerdict(index=1, supported=True, reason=""),
            ]), 900, 30)  # fmt: skip
        raise AssertionError(f"unexpected structured request for {schema}")

    def create(self, **kwargs: Any) -> Any:
        self.calls.append({"kind": "stream" if kwargs.get("stream") else "create", **kwargs})
        return stream_reply(list(ANSWER), 2000, 60)


@dataclass
class _Chat:
    completions: AutoCompletions


@dataclass
class _OpenAI:
    chat: _Chat


def auto_chat(rows: list[tuple[str, str, int, int]] | None = None) -> ChatClient:
    rows = [] if rows is None else rows

    def record(feature: str, model: str, tokens_in: int, tokens_out: int) -> Decimal:
        rows.append((feature, model, tokens_in, tokens_out))
        return Decimal("0.0001")

    return ChatClient(_OpenAI(_Chat(AutoCompletions())), record, sleep=lambda s: None)


class FakeJudge(Judge):
    """Fixed verdicts; `recall` of the reference sentences are found in the sources."""

    def __init__(self, recall: float = 1.0, model: str = "invented-judge") -> None:
        client = auto_chat()
        client.model = model
        super().__init__(client, max_tokens=500, relevancy_questions=3)
        self.recall = recall
        self.calls: list[str] = []

    def _spend(self, step: str) -> None:
        self.calls.append(step)
        self.spent.append(ChatUsage(10, 5, Decimal("0.00001")))

    def statements(self, question: str, answer: str) -> list[str]:
        self._spend("statements")
        return ["LoRA trains matrices.", "Weights frozen."]

    def faithfulness(self, statements: Sequence[str], sources: Sequence[Source]) -> list[Verdict]:
        self._spend("faithfulness")
        return [Verdict(i, True, "") for i, _ in enumerate(statements)]

    def relevancy(self, answer: str) -> tuple[list[str], bool]:
        self._spend("relevancy")
        return [], False

    def context_precision(self, question: str, reference: str,
                          sources: Sequence[Source]) -> list[Verdict]:  # fmt: skip
        self._spend("context_precision")
        return [Verdict(s.n, k == 0, "") for k, s in enumerate(sources)]

    def context_recall(self, question: str, reference_sentences: Sequence[str],
                       sources: Sequence[Source]) -> list[Verdict]:  # fmt: skip
        if not sources:
            return [Verdict(i, False, "") for i, _ in enumerate(reference_sentences)]
        self._spend("context_recall")
        found = round(self.recall * len(reference_sentences))
        return [Verdict(i, i < found, "") for i, _ in enumerate(reference_sentences)]

    def citations(self, items: Sequence[CitationItem], sources: Sequence[Source]) -> list[Verdict]:
        self._spend("citations")
        return [Verdict(k, True, "") for k in range(len(items))]


class FakeRelevancyEmbed:
    def embed(self, texts: Sequence[str], feature: str) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


@dataclass
class _EmbeddingItem:
    embedding: list[float]
    index: int


@dataclass
class _EmbeddingUsage:
    prompt_tokens: int


@dataclass
class _EmbeddingResponse:
    data: list[_EmbeddingItem]
    usage: _EmbeddingUsage


@dataclass
class FakeOpenAIEmbeddings:
    """`openai.OpenAI().embeddings`, answering with vectors of the size the settings ask
    for at the moment of the request; records (model, size asked, `dimensions`)."""

    requests: list[tuple[str, int, int | None]] = field(default_factory=list)

    @property
    def embeddings(self) -> "FakeOpenAIEmbeddings":
        return self

    def create(self, *, input: list[str], model: str, dimensions: int | None = None) -> Any:
        dim = get_settings().embedding_dim
        self.requests.append((model, dim, dimensions))
        return _EmbeddingResponse(
            [_EmbeddingItem([0.1] * dim, i) for i, _ in enumerate(input)], _EmbeddingUsage(7)
        )


@dataclass
class FakeSearch:
    """The chat's search: embeds the query when given a dense encoder, returns `hits`."""

    hits: list[SearchHit]
    calls: list[dict[str, Any]] = field(default_factory=list)

    def __call__(self, query: str, filters: Any, **kw: Any) -> list[SearchHit]:
        self.calls.append({"query": query, "mode": kw.get("mode"), "limit": kw.get("limit"),
                           "collection_in_settings": get_settings().qdrant_collection})  # fmt: skip
        if kw.get("dense") is not None:
            kw["dense"].embed([query], "embed-query")
        return list(self.hits)


@dataclass
class FakeRetrieve:
    """`app.rag.retrieve.retrieve`: records what each call got, returns `hits`."""

    hits: list[SearchHit]
    calls: list[dict[str, Any]] = field(default_factory=list)

    def __call__(self, query: str, filters: Any, **kw: Any) -> list[SearchHit]:
        self.calls.append(kw)
        if kw.get("dense") is not None:
            kw["dense"].embed([query], "embed-query")
        encoder = kw.get("cross_encoder")
        if kw["config"].rerank and encoder is not None:
            list(encoder.rerank(query, [h.text for h in self.hits], batch_size=16))
        return list(self.hits)


class FakeCrossEncoder:
    def __init__(self, key: RerankerKey) -> None:
        self.key = key
        self.calls = 0

    def rerank(self, query: str, documents: Iterable[str], batch_size: int = 64) -> list[float]:
        self.calls += 1
        return [1.0 for _ in documents]


@dataclass
class KeyedCrossEncoders:
    """One fake model per reranker key of the current settings (like get_cross_encoder)."""

    loads: list[RerankerKey] = field(default_factory=list)
    current: FakeCrossEncoder | None = None

    def get(self) -> FakeCrossEncoder:
        key = reranker_key()
        if self.current is None or self.current.key != key:
            self.loads.append(key)
            self.current = FakeCrossEncoder(key)
        return self.current

    def loaded(self) -> RerankerKey | None:
        return None if self.current is None else self.current.key


@dataclass
class MemoryStore:
    """A RunStore in memory (eval/runner.py)."""

    runs: dict[int, dict[str, Any]] = field(default_factory=dict)
    results: list[dict[str, Any]] = field(default_factory=list)
    checked: list[str] = field(default_factory=list)
    texts: dict[uuid.UUID, str] = field(default_factory=lambda: dict(REFERENCE_TEXTS))

    def check(self, questions: Sequence[SetQuestion], collection: str) -> None:
        self.checked.append(collection)

    def reference_texts(self, ids: Iterable[uuid.UUID]) -> dict[uuid.UUID, str]:
        return {i: self.texts[i] for i in ids if i in self.texts}

    def sync_items(self, questions: Sequence[SetQuestion]) -> dict[str, int]:
        return {q.item.id: n for n, q in enumerate(questions, start=1)}

    def start(self, name: str, config: dict[str, Any], git_sha: str | None) -> int:
        run_id = len(self.runs) + 1
        self.runs[run_id] = {"name": name, "config": config, "git_sha": git_sha,
                             "metrics": {"status": "running"}}  # fmt: skip
        return run_id

    def add(self, run_id: int, item_id: int, retrieved: Sequence[uuid.UUID], answer: str | None,
            metrics: dict[str, Any]) -> None:  # fmt: skip
        self.results.append({"run_id": run_id, "item_id": item_id, "retrieved": list(retrieved),
                             "answer": answer, "metrics": metrics})  # fmt: skip

    def finish(self, run_id: int, metrics: dict[str, Any], status: str) -> dict[str, Any]:
        self.runs[run_id]["metrics"] = {**metrics, "status": status, "spend_usd": "0"}
        return dict(self.runs[run_id]["metrics"])

    def of(self, run_id: int) -> list[dict[str, Any]]:
        return [r for r in self.results if r["run_id"] == run_id]

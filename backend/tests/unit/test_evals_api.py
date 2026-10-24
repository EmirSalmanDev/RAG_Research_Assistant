"""GET /evals/...: the read-only routes of the /evals page on a fake store (no database).

Every run shape stored so far is covered: a `make eval-retrieval` run (flat metrics, no
texts stored), an ablation answers run (per split, sources with texts), a selection."""

import uuid
from collections.abc import AsyncIterator, Iterator
from datetime import UTC, datetime
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.evals import ChunkInfo, ResultRecord, RunRecord, get_eval_store, run_kind
from app.main import app

REF = uuid.uuid5(uuid.NAMESPACE_URL, "ref")
OTHER = uuid.uuid5(uuid.NAMESPACE_URL, "other")
WHEN = datetime(2026, 10, 21, 12, 0, tzinfo=UTC)

LEGACY = RunRecord(
    17, "hybrid", {"name": "hybrid", "mode": "hybrid", "rerank": False, "collection": "chunks"},
    {"recall@8": 0.912, "mrr": 0.775, "ndcg@10": 0.794, "n_scored": 17, "n_unscored": 3,
     "latency_ms_p50": 30.1},
    "4a34a4d", WHEN,
)  # fmt: skip
ANSWERS = RunRecord(
    31, "ablation:a3-hybrid:answers",
    {"kind": "answers", "suite": {"kind": "ablation", "step": "retrieval", "id": "s1"},
     "variant": {"variant": "a3-hybrid", "label": "hybrid (RRF)"}},
    {"status": "complete", "spend_usd": "0.2",
     "all": {"n_questions": 2, "faithfulness": 0.75, "cost_per_answer_usd": "0.004"},
     "generated": {"n_questions": 2, "faithfulness": 0.75}},
    "abc123", WHEN,
)  # fmt: skip
SELECTION = RunRecord(40, "selection:retrieval", {"step": "retrieval"},
                      {"status": "complete", "winner": "a3-hybrid"}, "abc123", WHEN)  # fmt: skip

LEGACY_RESULT = ResultRecord(5, "Invented legacy question?", None, [REF], [OTHER, REF], None,
                             {"recall@8": 1.0, "mrr": 0.5, "golden_id": "q01"})  # fmt: skip
ANSWER_RESULT = ResultRecord(
    6, "Invented generated question?", "Invented reference.", [REF], [REF], "It does [1].",
    {"golden_id": "gen-001", "split": "generated", "type": "fact", "faithfulness": 1.0,
     "cost_usd": "0.0041", "latency_ms": 2100.0, "refusal": None, "error": None,
     "details": {"faithfulness": [{"statement": "It does.", "supported": True}]},
     "judge_errors": {"citation_accuracy": "JudgeError: x"},
     "sources": [{"n": 1, "chunk_id": str(REF), "arxiv_id": "2601.00001", "title": "Paper",
                  "section_path": "3 Method", "chunk_type": "section", "score": 0.03,
                  "text": "Invented stored text."}]},
)  # fmt: skip


class FakeStore:
    def __init__(self) -> None:
        self.looked_up: list[list[uuid.UUID]] = []

    def list_runs(self, limit: int) -> list[RunRecord]:
        return [SELECTION, ANSWERS, LEGACY][:limit]

    def get_run(self, run_id: int) -> RunRecord | None:
        return {r.id: r for r in (LEGACY, ANSWERS, SELECTION)}.get(run_id)

    def results(self, run_id: int) -> list[ResultRecord]:
        return {17: [LEGACY_RESULT], 31: [ANSWER_RESULT]}.get(run_id, [])

    def result(self, run_id: int, item_id: int) -> ResultRecord | None:
        return next((r for r in self.results(run_id) if r.item_id == item_id), None)

    def chunks(self, ids: list[uuid.UUID]) -> dict[uuid.UUID, ChunkInfo]:
        self.looked_up.append(ids)
        info = ChunkInfo(
            "2601.00002", "Other paper", "2 Background", "section", "Text from chunks."
        )
        return {i: info for i in ids}


@pytest.fixture
def store() -> Iterator[FakeStore]:
    fake = FakeStore()
    app.dependency_overrides[get_eval_store] = lambda: fake
    yield fake
    app.dependency_overrides.clear()


@pytest.fixture
async def client(store: FakeStore) -> AsyncIterator[AsyncClient]:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


async def test_the_run_list_has_every_shape_with_headline_metrics(client: AsyncClient) -> None:
    r = await client.get("/evals/runs")
    assert r.status_code == 200
    runs: list[dict[str, Any]] = r.json()["runs"]
    assert [(x["id"], x["kind"]) for x in runs] == [
        (40, "selection"), (31, "answers"), (17, "retrieval"),
    ]  # fmt: skip
    legacy, answers = runs[2], runs[1]
    assert legacy["metrics"]["recall@8"] == 0.912 and legacy["n_questions"] == 20
    assert legacy["status"] == "complete" and legacy["splits"] == []
    assert answers["metrics"] == {
        "n_questions": 2.0,
        "faithfulness": 0.75,
        "cost_per_answer_usd": 0.004,
    }
    assert (answers["suite"], answers["step"], answers["variant"]) == (
        "ablation",
        "retrieval",
        "a3-hybrid",
    )
    assert answers["splits"] == ["all", "generated"]


async def test_one_run_carries_its_configuration_and_every_metric(client: AsyncClient) -> None:
    r = await client.get("/evals/runs/31")
    body = r.json()
    assert body["config"]["variant"]["label"] == "hybrid (RRF)"
    assert body["all_metrics"]["generated"]["faithfulness"] == 0.75
    assert (await client.get("/evals/runs/999")).status_code == 404


async def test_results_are_one_row_of_scores_per_question(client: AsyncClient) -> None:
    body = (await client.get("/evals/runs/31/results")).json()
    assert body["kind"] == "answers"
    (only,) = body["results"]
    assert (only["question_id"], only["split"], only["outcome"]) == (
        "gen-001",
        "generated",
        "answered",
    )
    assert only["scores"] == {"faithfulness": 1.0, "cost_usd": 0.0041, "latency_ms": 2100.0}
    legacy = (await client.get("/evals/runs/17/results")).json()["results"][0]
    assert legacy["outcome"] == "retrieved" and legacy["scores"]["mrr"] == 0.5


async def test_a_question_shows_its_chunks_answer_scores_and_verdicts(
    client: AsyncClient, store: FakeStore
) -> None:
    body = (await client.get("/evals/runs/31/results/6")).json()
    (chunk,) = body["retrieved"]
    assert chunk["text"] == "Invented stored text." and chunk["n"] == 1 and chunk["is_reference"]
    assert chunk["abs_url"] == "https://arxiv.org/abs/2601.00001"
    assert body["answer"] == "It does [1]." and body["reference_answer"] == "Invented reference."
    assert body["judge"]["faithfulness"][0]["supported"] is True
    assert body["judge_errors"] == {"citation_accuracy": "JudgeError: x"}
    assert body["cost_usd"] == 0.0041
    assert store.looked_up == [[]]  # the texts were stored with the result: no lookup


async def test_an_older_run_without_texts_reads_them_from_the_chunks_table(
    client: AsyncClient, store: FakeStore
) -> None:
    body = (await client.get("/evals/runs/17/results/5")).json()
    assert [c["rank"] for c in body["retrieved"]] == [1, 2]
    assert [c["is_reference"] for c in body["retrieved"]] == [False, True]
    assert body["retrieved"][0]["text"] == "Text from chunks."
    assert store.looked_up == [[OTHER, REF]]
    assert (await client.get("/evals/runs/17/results/99")).status_code == 404


def test_a_run_on_another_index_never_borrows_texts_from_the_chunks_table() -> None:
    from app.api.evals import needs_lookup

    variant = RunRecord(50, "ablation:a7-c256:retrieval",
                        {"kind": "retrieval", "effective": {"collection": "chunks__eval__c256"}},
                        {}, None, WHEN)  # fmt: skip
    record = ResultRecord(1, "Q?", None, [], [REF], None, {})
    assert needs_lookup(variant, record) == []
    assert run_kind(variant) == "retrieval"
    assert run_kind(RunRecord(3, "answers:default", {}, {}, None, WHEN)) == "answers"


async def test_no_reference_badge_on_another_chunking(client: AsyncClient) -> None:
    """Under 256-token chunks the id of the 512-token reference names another text."""
    from app.api.evals import retrieved

    other = RunRecord(51, "ablation:a7-c256:retrieval",
                      {"kind": "retrieval", "same_chunking": False}, {}, None, WHEN)  # fmt: skip
    hits = {"hits": [{"chunk_id": str(REF), "text": "Other chunk text."}]}
    record = ResultRecord(1, "Q?", None, [REF], [REF], None, hits)
    (chunk,) = retrieved(other, record, {})
    assert chunk.is_reference is False and chunk.text == "Other chunk text."
    (same,) = retrieved(ANSWERS, record, {})
    assert same.is_reference is True

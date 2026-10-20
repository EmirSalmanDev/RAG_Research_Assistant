"""POST /library/review with a fake model: status codes and shapes, no database, no OpenAI."""

from collections.abc import AsyncIterator, Iterator

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.review import get_review_deps
from app.features.review import (
    DraftOut,
    DraftParagraph,
    DraftSection,
    DraftSentence,
    ReviewDeps,
)
from app.llm.client import ChatClient
from app.main import app
from tests.chat_fakes import fake_chat, parsed_reply
from tests.review_fakes import FakeStore, collection
from tests.tracing_fakes import RecordingTracer


def _deps(replies: list[object], papers: int = 3) -> ReviewDeps:
    chat, _, _ = fake_chat(replies)
    return ReviewDeps(chat, FakeStore({"Thesis": collection(papers)}), RecordingTracer())


@pytest.fixture
def use() -> Iterator[list[ReviewDeps]]:
    holder: list[ReviewDeps] = []
    app.dependency_overrides[get_review_deps] = lambda: holder[-1]
    yield holder
    app.dependency_overrides.clear()


@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


OK = DraftOut(
    sections=[
        DraftSection(
            heading="Widgets",
            paragraphs=[
                DraftParagraph(
                    sentences=[
                        DraftSentence(text="Authora studied it.", kind="claim", source_numbers=[1]),
                        DraftSentence(text="Nobody checked.", kind="claim", source_numbers=[]),
                    ]
                )
            ],
        )
    ]
)


async def test_a_review_returns_the_draft_references_and_counts(
    client: AsyncClient, use: list[ReviewDeps]
) -> None:
    use.append(_deps([parsed_reply(OK, 700, 90)]))
    r = await client.post("/library/review", json={"collection": "Thesis", "focus": "widgets"})
    assert r.status_code == 200
    body = r.json()
    [sentence] = body["sections"][0]["paragraphs"][0]["sentences"]
    assert sentence == {"text": "Authora studied it.", "kind": "claim", "source_numbers": [1]}
    assert body["counts"]["dropped_uncited"] == 1 and body["counts"]["papers"] == 3
    assert body["references"][0]["bibtex_key"] == "authora2020invented"
    assert body["references"][0]["evidence"] == "analysis"
    assert (body["tokens_in"], body["tokens_out"]) == (700, 90) and body["cost_usd"] > 0
    assert body["prompt_versions"] == {"write": "review_write_v1"} and body["focus"] == "widgets"


async def test_refusals_come_before_any_model_call(
    client: AsyncClient, use: list[ReviewDeps]
) -> None:
    deps = _deps([], papers=1)
    use.append(deps)
    r = await client.post("/library/review", json={"collection": "Thesis"})
    assert r.status_code == 422 and "at least 2 papers" in r.json()["detail"]
    assert (await client.post("/library/review", json={"collection": "Nope"})).status_code == 404
    for bad in (
        {"collection": ""},
        {"collection": "Thesis", "focus": "x" * 201},
        {"collection": "Thesis", "max_papers": 0},
    ):
        assert (await client.post("/library/review", json=bad)).status_code == 422
    r = await client.post("/library/review", json={"collection": "Thesis", "max_papers": 61})
    assert r.status_code == 422 and "between 1 and 60" in r.json()["detail"]


async def test_a_missing_key_is_503_and_a_model_failure_is_502(
    client: AsyncClient, use: list[ReviewDeps]
) -> None:
    keyless = _deps([])
    keyless.chat = ChatClient()  # no key in the tests
    use.append(keyless)
    r = await client.post("/library/review", json={"collection": "Thesis"})
    assert r.status_code == 503
    use.append(_deps([parsed_reply(None)]))
    r = await client.post("/library/review", json={"collection": "Thesis"})
    assert r.status_code == 502

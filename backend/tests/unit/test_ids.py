"""The id of an uploaded document and the arXiv link that only arXiv papers have."""

import uuid

from app.ids import abs_url, is_local_id
from app.rag.citations import Source, source_object
from app.rag.search import SearchHit


def test_local_ids_are_recognised_and_have_no_arxiv_link() -> None:
    assert is_local_id("local-0123456789ab") and not is_local_id("2601.00001")
    assert not is_local_id("hep-th/9901001")
    assert abs_url("local-0123456789ab") is None
    assert abs_url("2601.00001") == "https://arxiv.org/abs/2601.00001"
    assert abs_url("hep-th/9901001") == "https://arxiv.org/abs/hep-th/9901001"


def test_a_chat_source_of_an_uploaded_document_carries_no_link() -> None:
    hit = SearchHit(
        chunk_id=uuid.UUID(int=1), arxiv_id="local-0123456789ab", title="Uploaded",
        section_path="Abstract", chunk_type="abstract",
        published_at="2026-01-01T00:00:00+00:00", text="Invented.", score=1.0,
    )  # fmt: skip
    assert source_object(Source(1, hit))["abs_url"] is None

"""Collections, items, status, notes and the digest's "Saved" on a scratch database.
All papers are invented."""

import uuid

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import Chunk, LibraryCollection, LibraryItem, Paper
from app.features import digest, library
from app.features.bibtex import build_entries, render_bibtex
from app.features.digest import SAVED_COLLECTION, apply_action, build_digest
from app.features.library import (
    UNSET,
    CollectionExists,
    CollectionNotFound,
    InvalidInput,
    ItemExists,
    ItemNotFound,
    PaperNotHeld,
)
from tests.bib_validate import keys, validate
from tests.digest_fakes import NOW, FakeVectors, make_interest, make_paper

pytestmark = pytest.mark.integration


@pytest.fixture
def s(migrated_session: Session) -> Session:
    make_paper(migrated_session, "2610.00001", age_days=3, title="Widgets and Gadgets")
    make_paper(migrated_session, "2610.00002", age_days=2, title="More Widgets")
    make_paper(migrated_session, "2610.00003", age_days=1, title="Even More Widgets")
    return migrated_session


def test_a_collection_can_be_created_empty_and_is_listed(s: Session) -> None:
    info = library.create_collection(s, "  Thesis   chapter 2 ", "Related work")
    assert (info.name, info.description, info.paper_count) == (
        "Thesis chapter 2",
        "Related work",
        0,
    )
    assert [c.name for c in library.list_collections(s)] == ["Thesis chapter 2"]
    assert library.list_items(s, "Thesis chapter 2") == []


def test_a_name_is_unique_and_validated(s: Session) -> None:
    library.create_collection(s, "A")
    with pytest.raises(CollectionExists):
        library.create_collection(s, "A")
    for bad in ("", "   ", "x" * 101, "bad\x00name"):
        with pytest.raises(InvalidInput):
            library.create_collection(s, bad)
    with pytest.raises(InvalidInput):
        library.create_collection(s, "ok", "d" * 501)


def test_the_list_is_the_union_of_the_table_and_the_items(s: Session) -> None:
    library.create_collection(s, "Empty")
    library.add_item(s, "2610.00001", "Saved")  # no row in library_collections
    library.add_item(s, "2610.00002", "Saved")
    listing = {c.name: c for c in library.list_collections(s)}
    assert set(listing) == {"Empty", "Saved"}
    assert listing["Saved"].paper_count == 2 and listing["Saved"].created_at is None
    assert listing["Empty"].paper_count == 0 and listing["Empty"].created_at is not None


def test_adding_an_item_by_id_version_or_link_is_idempotent(s: Session) -> None:
    item, created = library.add_item(s, "https://arxiv.org/abs/2610.00001v3", "Thesis")
    assert created and (item.arxiv_id, item.status, item.notes) == ("2610.00001", "to_read", None)
    assert item.evidence == "abstract_only" and item.abs_url == "https://arxiv.org/abs/2610.00001"
    again, created = library.add_item(s, "arXiv:2610.00001", "Thesis")
    assert not created and again == item
    library.add_item(s, "https://arxiv.org/pdf/2610.00002v1.pdf", "Thesis")
    assert [i.arxiv_id for i in library.list_items(s, "Thesis")] == ["2610.00001", "2610.00002"]
    assert s.scalar(select(func.count()).select_from(LibraryItem)) == 2


def test_an_unknown_paper_or_a_bad_reference_is_refused(s: Session) -> None:
    with pytest.raises(PaperNotHeld) as err:
        library.add_item(s, "2610.99999", "Thesis")
    assert "Ingest the paper first" in str(err.value)
    for bad in ("hello", "https://example.com/abs/2610.00001", "", "2610.1"):
        with pytest.raises(InvalidInput):
            library.add_item(s, bad, "Thesis")
    with pytest.raises(InvalidInput):
        library.add_item(s, "2610.00001", "Thesis", status="finished")
    assert not library.collection_exists(s, "Thesis")  # nothing was created by the failures


def test_an_old_style_id_is_accepted(s: Session) -> None:
    make_paper(s, "hep-th/9901001", age_days=5)
    item, _ = library.add_item(s, "https://arxiv.org/abs/hep-th/9901001v2", "Old")
    assert item.arxiv_id == "hep-th/9901001"


def test_status_and_notes_change_and_blank_notes_clear(s: Session) -> None:
    library.add_item(s, "2610.00001", "Thesis")
    item = library.update_item(s, "2610.00001", "Thesis", status="reading", notes="  check eq. 3 ")
    assert (item.status, item.notes) == ("reading", "check eq. 3")
    item = library.update_item(s, "2610.00001", "Thesis", notes="")
    assert (item.status, item.notes) == ("reading", None)
    item = library.update_item(s, "2610.00001", "Thesis", status="archived")
    assert item.status == "archived"
    for bad in ("done", ""):
        with pytest.raises(InvalidInput):
            library.update_item(s, "2610.00001", "Thesis", status=bad)
    with pytest.raises(InvalidInput):
        library.update_item(s, "2610.00001", "Thesis", notes="n" * 10_001)
    with pytest.raises(ItemNotFound):
        library.update_item(s, "2610.00002", "Thesis", status="read")


def test_an_item_moves_to_another_collection_unless_it_is_there_already(s: Session) -> None:
    library.add_item(s, "2610.00001", "A")
    library.add_item(s, "2610.00001", "B")
    library.add_item(s, "2610.00002", "A")
    library.update_item(s, "2610.00001", "A", notes="keep me")
    with pytest.raises(ItemExists):
        library.update_item(s, "2610.00001", "A", new_collection="B")
    moved = library.update_item(s, "2610.00002", "A", new_collection="C")
    assert moved.collection == "C" and moved.arxiv_id == "2610.00002"
    assert [i.arxiv_id for i in library.list_items(s, "A")] == ["2610.00001"]
    assert library.list_items(s, "A")[0].notes == "keep me"
    assert library.update_item(s, "2610.00001", "A", new_collection="A").collection == "A"


def test_removing_an_item_keeps_the_paper(s: Session) -> None:
    library.add_item(s, "2610.00001", "A")
    library.remove_item(s, "2610.00001", "A")
    with pytest.raises(ItemNotFound):
        library.remove_item(s, "2610.00001", "A")
    assert s.get(Paper, "2610.00001") is not None


def test_a_rename_moves_the_items_in_one_transaction(s: Session) -> None:
    library.create_collection(s, "Old", "About things")
    library.add_item(s, "2610.00001", "Old", status="read")
    info = library.update_collection(s, "Old", new_name="New")
    assert (info.name, info.description, info.paper_count) == ("New", "About things", 1)
    assert not library.collection_exists(s, "Old")
    assert library.list_items(s, "New")[0].status == "read"
    library.create_collection(s, "Taken")
    with pytest.raises(CollectionExists):
        library.update_collection(s, "New", new_name="Taken")
    with pytest.raises(CollectionNotFound):
        library.update_collection(s, "Nope", new_name="X")
    assert library.update_collection(s, "New", description=None).description is None


def test_an_implicit_collection_can_be_renamed_and_described(s: Session) -> None:
    library.add_item(s, "2610.00001", SAVED_COLLECTION)
    info = library.update_collection(s, SAVED_COLLECTION, description="From the digest")
    assert info.description == "From the digest" and info.paper_count == 1
    info = library.update_collection(
        s, SAVED_COLLECTION, new_name="Digest saves", description=UNSET
    )
    assert info.name == "Digest saves" and info.description == "From the digest"
    assert s.scalar(select(func.count()).select_from(LibraryCollection)) == 1


def test_deleting_a_collection_removes_its_items_but_not_the_papers(s: Session) -> None:
    library.create_collection(s, "Gone")
    library.add_item(s, "2610.00001", "Gone")
    library.add_item(s, "2610.00001", "Kept")
    assert library.delete_collection(s, "Gone") == 1
    assert not library.collection_exists(s, "Gone") and library.collection_exists(s, "Kept")
    assert s.get(Paper, "2610.00001") is not None
    with pytest.raises(CollectionNotFound):
        library.delete_collection(s, "Gone")
    assert library.delete_collection(s, "Kept") == 1  # an implicit one has no row to delete


def test_listing_the_items_of_an_unknown_collection_is_not_found(s: Session) -> None:
    with pytest.raises(CollectionNotFound):
        library.list_items(s, "Nope")


def test_evidence_follows_the_chunks_and_the_id(s: Session) -> None:
    make_paper(s, "2610.00004", age_days=4)
    chunk = Chunk(
        id=uuid.UUID(int=1), arxiv_id="2610.00004", chunk_type="section", section_path="Intro",
        ord=1, text="Invented.", token_count=2,
    )  # fmt: skip
    upload = Paper(
        arxiv_id="local-0123456789ab", title="Uploaded", abstract="Invented.", published_at=NOW,
        authors=[], categories=[],
    )  # fmt: skip
    s.add_all([chunk, upload])
    s.commit()
    for paper_id in ("2610.00001", "2610.00004", "local-0123456789ab"):
        library.add_item(s, paper_id, "E")
    by_id = {i.arxiv_id: i for i in library.list_items(s, "E")}
    assert by_id["2610.00001"].evidence == "abstract_only"
    assert by_id["2610.00004"].evidence == "full_text"
    assert by_id["local-0123456789ab"].evidence == "uploaded"
    assert by_id["local-0123456789ab"].abs_url is None


def test_the_bib_export_follows_the_stable_order_and_has_unique_keys(s: Session) -> None:
    for paper_id in ("2610.00003", "2610.00001", "2610.00002"):
        library.add_item(s, paper_id, "Bib")
    papers = library.collection_bib_papers(s, "Bib")
    assert [p.arxiv_id for p in papers] == ["2610.00001", "2610.00002", "2610.00003"]
    text = render_bibtex(build_entries(papers))
    assert validate(text) == [] and len(set(keys(text))) == 3
    assert text == render_bibtex(build_entries(library.collection_bib_papers(s, "Bib")))


def test_digest_save_and_undo_still_use_the_saved_collection(s: Session) -> None:
    """Regression: the M7 Save / Undo flow writes library_items; it must keep working with
    the library (a collection with no row, a note or a status change)."""
    interest = make_interest(s, "Alpha", embedding=[1.0, 0.0])
    vectors = FakeVectors({"2610.00001": [1.0, 0.0], "2610.00002": [0.0, 1.0]})
    result = build_digest(s, vectors, now=NOW)
    assert interest.id is not None and result.entries
    day = digest.today_in_zone(NOW)
    apply_action(s, day, "2610.00001", "save")
    s.commit()
    assert [c.name for c in library.list_collections(s)] == [SAVED_COLLECTION]
    item = library.list_items(s, SAVED_COLLECTION)[0]
    assert (item.arxiv_id, item.status) == ("2610.00001", "to_read")

    apply_action(s, day, "2610.00001", "undo")  # a plain save is removed again
    s.commit()
    assert s.scalar(select(func.count()).select_from(LibraryItem)) == 0

    apply_action(s, day, "2610.00001", "save")
    library.update_item(s, "2610.00001", SAVED_COLLECTION, notes="mine")  # the owner touched it
    s.commit()
    apply_action(s, day, "2610.00001", "undo")  # ... so Undo leaves the item alone
    s.commit()
    assert library.get_item(s, "2610.00001", SAVED_COLLECTION).notes == "mine"

"""The daily digest: score papers against the interests, build the day's list, and let
Save and Dismiss teach the weights. The rules are in docs/adr/0009.

Three entry points, each a thin step of the nightly chain (app/features/tasks.py):

- `ensure_interest_embeddings`: the only paid call in this feature. Interests whose text
  changed (or that were never embedded) are embedded in one request; the API and the
  pages never do this.
- `score_window`: scores every paper published within the window against the active
  interests (pure function in app/features/relevance.py) and stores `relevance_score`.
- `build_digest`: ranks the candidates, keeps the top `digest_size` and writes one
  `digest_entries` row each. Calling it again for the same date changes nothing (a
  `rebuild` replaces only the entries nobody acted on).

`apply_action` is the Save / Dismiss / Undo state machine. Reading the digest
(`load_digest`) never builds anything and never calls an API.
"""

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from typing import Any, Literal, Protocol
from zoneinfo import ZoneInfo

from qdrant_client import QdrantClient
from sqlalchemy import bindparam, delete, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.models import DigestEntry, Interest, LibraryItem, Paper, PaperAnalysis
from app.features import app_settings
from app.features import interests as interest_repo
from app.features.relevance import InterestVector, PaperFeatures, Score, rank_key, score_paper
from app.ids import LOCAL_PREFIX
from app.ingest.chunk import ABSTRACT_ORD, chunk_id
from app.ingest.embed import DenseEncoder
from app.ingest.index import DENSE_VECTOR, LOOKUP_BATCH, get_qdrant

logger = logging.getLogger(__name__)

SAVED_COLLECTION = "Saved"  # library collection that Save adds the paper to
SAVED_STATUS = "to_read"
INTEREST_EMBED_FEATURE = "embed-interest"  # llm_usage.feature of the interest embedding

Action = Literal["save", "dismiss", "undo"]


# --- abstract vectors -------------------------------------------------------------------------


class AbstractVectors(Protocol):
    """Where the paper's abstract vector comes from (Qdrant; tests pass a fake)."""

    def get(self, papers: Sequence[tuple[str, int]]) -> dict[str, list[float]]:
        """{arxiv_id: dense vector} for (arxiv_id, version) pairs; papers whose abstract
        point is missing are left out."""
        ...


class QdrantAbstractVectors:
    """Reads the stored dense vector of each paper's abstract point (its id is the
    deterministic uuid5 of arxiv_id, version and ord 0). Free: no embedding request."""

    def __init__(self, client: QdrantClient | None = None, collection: str | None = None) -> None:
        self._client = client
        self._collection = collection or get_settings().qdrant_collection

    def get(self, papers: Sequence[tuple[str, int]]) -> dict[str, list[float]]:
        client = self._client or get_qdrant()
        owner = {str(chunk_id(a, v, ABSTRACT_ORD)): a for a, v in papers}
        ids = list(owner)
        found: dict[str, list[float]] = {}
        for start in range(0, len(ids), LOOKUP_BATCH):
            records = client.retrieve(
                self._collection,
                ids=ids[start : start + LOOKUP_BATCH],
                with_payload=False,
                with_vectors=[DENSE_VECTOR],
            )
            for record in records:
                vector = record.vector
                if isinstance(vector, dict) and vector.get(DENSE_VECTOR) is not None:
                    found[owner[str(record.id)]] = list(vector[DENSE_VECTOR])  # type: ignore[arg-type]
        return found


# --- time -------------------------------------------------------------------------------------


def digest_zone() -> ZoneInfo:
    return ZoneInfo(get_settings().digest_timezone)


def today_in_zone(now: datetime | None = None) -> date:
    return (now or datetime.now(UTC)).astimezone(digest_zone()).date()


def reference_time(digest_date: date, now: datetime) -> datetime:
    """The moment a digest is built "as of": now for today's digest, the end of that day
    for an earlier date (so `--date` rebuilds a past day from the papers it could see)."""
    end_of_day = datetime.combine(digest_date + timedelta(days=1), time.min, tzinfo=digest_zone())
    return min(now, end_of_day.astimezone(UTC))


# --- interests: embeddings ----------------------------------------------------------------------


def active_interests(session: Session) -> list[Interest]:
    return list(
        session.scalars(select(Interest).where(Interest.is_active.is_(True)).order_by(Interest.id))
    )


def ensure_interest_embeddings(session: Session, dense: DenseEncoder) -> int:
    """Embed the active interests that have no vector for their current text, in one
    request. Returns how many were embedded (0 means no API call was made). The caller
    commits."""
    model = get_settings().embedding_model
    stale = [i for i in active_interests(session) if not interest_repo.has_current_embedding(i)]
    if not stale:
        return 0
    texts = [interest_repo.embedding_text(i.name, i.description, i.keywords) for i in stale]
    vectors = dense.embed(texts, INTEREST_EMBED_FEATURE)
    for interest, text, vector in zip(stale, texts, vectors, strict=True):
        interest.embedding = vector
        interest.embedding_model = model
        interest.embedding_hash = interest_repo.text_hash(model, text)
    session.flush()
    return len(stale)


def _interest_vectors(interests: Sequence[Interest]) -> list[InterestVector]:
    # An interest whose stored vector is stale (its text changed since) is still used: the
    # old vector is close to the new meaning, and the next scoring run replaces it.
    return [
        InterestVector(
            id=i.id, weight=i.weight, keywords=i.keywords, categories=i.categories,
            embedding=i.embedding,
        )
        for i in interests
    ]  # fmt: skip


# --- scoring ----------------------------------------------------------------------------------


@dataclass
class ScoreRun:
    scores: dict[str, Score]
    missing_vectors: int = 0


def _features(paper: Paper, vector: Sequence[float] | None) -> PaperFeatures:
    return PaperFeatures(
        arxiv_id=paper.arxiv_id, title=paper.title, abstract=paper.abstract,
        categories=paper.categories, hf_upvotes=paper.hf_upvotes,
        published_at=paper.published_at, vector=vector,
    )  # fmt: skip


def score_papers(
    papers: Sequence[Paper], interests: Sequence[InterestVector], vectors: AbstractVectors
) -> ScoreRun:
    """Score the papers. With no interest the vectors are not even read (nothing to compare)."""
    run = ScoreRun(scores={})
    found: dict[str, list[float]] = {}
    if interests:
        found = vectors.get([(p.arxiv_id, p.version) for p in papers])
        run.missing_vectors = sum(1 for p in papers if p.arxiv_id not in found)
    for paper in papers:
        run.scores[paper.arxiv_id] = score_paper(
            _features(paper, found.get(paper.arxiv_id)), interests
        )
    return run


def window_papers(session: Session, reference: datetime) -> list[Paper]:
    """Papers published within `digest_window_days` before `reference`."""
    start = reference - timedelta(days=get_settings().digest_window_days)
    return list(
        session.scalars(
            select(Paper)
            .where(Paper.published_at >= start, Paper.published_at <= reference)
            .where(~Paper.arxiv_id.startswith(LOCAL_PREFIX))  # uploads are not new arXiv papers
            .order_by(Paper.arxiv_id)
        )
    )


def _store_scores(session: Session, scores: dict[str, Score]) -> None:
    # updated_at is left alone: a new score is not a change to the paper.
    stmt = (
        update(Paper)
        .where(Paper.arxiv_id == bindparam("b_id"))
        .values(relevance_score=bindparam("b_score"), updated_at=Paper.updated_at)
    )
    rows = [{"b_id": arxiv_id, "b_score": s.score} for arxiv_id, s in scores.items()]
    if rows:
        session.connection().execute(stmt, rows)


def score_window(
    session: Session, vectors: AbstractVectors, now: datetime | None = None
) -> dict[str, Any]:
    """Step 4 of the nightly chain: `relevance_score` for every paper in the window.
    Idempotent: the same interests and papers give the same scores."""
    now = now or datetime.now(UTC)
    interests = _interest_vectors(active_interests(session))
    papers = window_papers(session, now)
    run = score_papers(papers, interests, vectors)
    _store_scores(session, run.scores)
    return {
        "papers": len(papers), "interests": len(interests), "missing_vectors": run.missing_vectors,
        "interests_without_vector": sum(1 for i in interests if i.embedding is None),
    }  # fmt: skip


# --- the digest -------------------------------------------------------------------------------


@dataclass
class PlannedEntry:
    arxiv_id: str
    rank: int
    score: float
    best_interest_id: int | None
    published_at: datetime
    title: str = ""
    status: str = "new"


@dataclass
class BuildResult:
    digest_date: date
    status: Literal["built", "exists", "empty"]
    candidates: int = 0
    kept: int = 0  # entries nobody-acted-on rebuild left alone
    entries: list[PlannedEntry] = field(default_factory=list)
    scores_missing_vectors: int = 0

    def counts(self) -> dict[str, Any]:
        return {
            "digest_date": self.digest_date.isoformat(), "status": self.status,
            "candidates": self.candidates, "entries": len(self.entries), "kept": self.kept,
            "missing_vectors": self.scores_missing_vectors,
        }  # fmt: skip


def build_digest(
    session: Session,
    vectors: AbstractVectors,
    *,
    digest_date: date | None = None,
    now: datetime | None = None,
    rebuild: bool = False,
    dry_run: bool = False,
) -> BuildResult:
    """Build one day's digest (default: today in the digest time zone).

    Candidates are papers published within the window that were not dismissed and were
    not in an earlier digest. They are scored again here (free: Qdrant reads only, no
    embedding), so the digest never depends on a stale `relevance_score`. The top
    `digest_size` become entries, ranked by score, then newer first, then arXiv id.

    A date that already has entries is left alone (`status="exists"`), so a second run of
    the nightly chain changes nothing. `rebuild=True` replaces only the entries that are
    still `new`; saved and dismissed entries stay and count toward the size.
    `dry_run` computes the result and writes nothing.
    """
    now = now or datetime.now(UTC)
    digest_date = digest_date or today_in_zone(now)
    size = app_settings.load(session).digest_size
    existing = list(
        session.scalars(select(DigestEntry).where(DigestEntry.digest_date == digest_date))
    )
    if existing and not rebuild:
        listed = [
            PlannedEntry(
                e.arxiv_id, e.rank, e.score, e.best_interest_id, _published(session, e.arxiv_id),
                title=_title(session, e.arxiv_id), status=e.status,
            )
            for e in sorted(existing, key=lambda e: e.rank)
        ]  # fmt: skip
        return BuildResult(digest_date, "exists", kept=len(existing), entries=listed)
    kept = [e for e in existing if e.status != "new"]

    # Never again: dismissed papers and papers of earlier digests. The entries of this
    # date that are kept stay as they are; its `new` ones are replaced and may come back.
    excluded = set(
        session.scalars(
            select(DigestEntry.arxiv_id).where(
                (DigestEntry.status == "dismissed")
                | (DigestEntry.digest_date < digest_date)
                | ((DigestEntry.digest_date == digest_date) & (DigestEntry.status != "new"))
            )
        )
    )
    interests = _interest_vectors(active_interests(session))
    papers = [p for p in window_papers(session, reference_time(digest_date, now))
              if p.arxiv_id not in excluded]  # fmt: skip
    run = score_papers(papers, interests, vectors)
    ranked = sorted(
        papers, key=lambda p: rank_key(p.arxiv_id, run.scores[p.arxiv_id].score, p.published_at)
    )[: max(0, size - len(kept))]

    # kept entries and fresh ones, merged and numbered together by the same key
    merged: list[PlannedEntry] = [
        PlannedEntry(e.arxiv_id, 0, e.score, e.best_interest_id, _published(session, e.arxiv_id),
                     status=e.status)
        for e in kept
    ]  # fmt: skip
    merged += [
        PlannedEntry(p.arxiv_id, 0, run.scores[p.arxiv_id].score,
                     run.scores[p.arxiv_id].best_interest_id, p.published_at, title=p.title)
        for p in ranked
    ]  # fmt: skip
    merged.sort(key=lambda e: rank_key(e.arxiv_id, e.score, e.published_at))
    for rank, entry in enumerate(merged, start=1):
        entry.rank = rank
    result = BuildResult(
        digest_date, "built" if merged else "empty", candidates=len(papers), kept=len(kept),
        entries=merged, scores_missing_vectors=run.missing_vectors,
    )  # fmt: skip
    if dry_run:
        return result

    _store_scores(session, run.scores)
    session.execute(
        delete(DigestEntry).where(
            DigestEntry.digest_date == digest_date, DigestEntry.status == "new"
        )
    )
    for entry in merged:
        if entry.status == "new":
            session.add(
                DigestEntry(
                    digest_date=digest_date,
                    arxiv_id=entry.arxiv_id,
                    rank=entry.rank,
                    score=entry.score,
                    best_interest_id=entry.best_interest_id,
                )  # fmt: skip
            )
        else:  # a kept entry only gets its new rank
            session.execute(
                update(DigestEntry)
                .where(
                    DigestEntry.digest_date == digest_date, DigestEntry.arxiv_id == entry.arxiv_id
                )
                .values(rank=entry.rank)
            )
    session.flush()
    return result


def _title(session: Session, arxiv_id: str) -> str:
    return session.scalar(select(Paper.title).where(Paper.arxiv_id == arxiv_id)) or ""


def _published(session: Session, arxiv_id: str) -> datetime:
    value = session.scalar(select(Paper.published_at).where(Paper.arxiv_id == arxiv_id))
    assert value is not None  # digest_entries.arxiv_id is a foreign key to papers
    return value


# --- feedback: Save, Dismiss, Undo --------------------------------------------------------------


class EntryNotFoundError(LookupError):
    """No digest entry for this date and paper."""


def _clamp(weight: float) -> float:
    s = get_settings()
    return round(
        min(s.interest_weight_max, max(s.interest_weight_min, weight)), s.interest_weight_decimals
    )


def _move_weight(session: Session, interest_id: int | None, delta: float) -> float:
    """Add `delta` to an interest's weight (clamped to the limits) and return the change
    that was really made, which is what the entry records. 0 when the interest is gone."""
    if interest_id is None:
        return 0.0
    interest = session.scalar(select(Interest).where(Interest.id == interest_id).with_for_update())
    if interest is None:
        return 0.0
    before = interest.weight
    interest.weight = _clamp(before + delta)
    return round(interest.weight - before, get_settings().interest_weight_decimals)


def _remove_saved_library_item(session: Session, arxiv_id: str) -> None:
    # Only if the owner has not touched it (still "to_read", no notes).
    session.execute(
        delete(LibraryItem).where(
            LibraryItem.arxiv_id == arxiv_id, LibraryItem.collection == SAVED_COLLECTION,
            LibraryItem.status == SAVED_STATUS, LibraryItem.notes.is_(None),
        )
    )  # fmt: skip


def apply_action(
    session: Session, digest_date: date, arxiv_id: str, action: Action, now: datetime | None = None
) -> DigestEntry:
    """Move an entry between new, saved and dismissed.

    - save / dismiss from `new`: nudge the weight of the entry's best interest by the
      configured amount (clamped) and remember the change really made in `weight_delta`;
      Save also puts the paper in the library collection "Saved".
    - the same action again: nothing changes, so a repeated click never counts twice.
    - undo: gives back exactly `weight_delta` and returns the entry to `new`.
    - save from dismissed (or the reverse): undo first, then the new action.
    The caller commits; the entry row is locked while the weights move.
    """
    s = get_settings()
    entry = session.scalar(
        select(DigestEntry)
        .where(DigestEntry.digest_date == digest_date, DigestEntry.arxiv_id == arxiv_id)
        .with_for_update()
    )
    if entry is None:
        raise EntryNotFoundError(f"{arxiv_id} is not in the digest of {digest_date}")
    target = {"save": "saved", "dismiss": "dismissed", "undo": "new"}[action]
    if entry.status == target:
        return entry
    if entry.status != "new":  # leave the current state first
        _move_weight(session, entry.best_interest_id, -entry.weight_delta)
        if entry.status == "saved":
            _remove_saved_library_item(session, arxiv_id)
        entry.weight_delta = 0.0
        entry.status = "new"
        entry.actioned_at = None
    if target != "new":
        delta = s.interest_save_delta if target == "saved" else s.interest_dismiss_delta
        entry.weight_delta = _move_weight(session, entry.best_interest_id, delta)
        entry.status = target
        entry.actioned_at = now or datetime.now(UTC)
        if target == "saved":
            session.execute(
                pg_insert(LibraryItem)
                .values(arxiv_id=arxiv_id, collection=SAVED_COLLECTION, status=SAVED_STATUS)
                .on_conflict_do_nothing()
            )
    session.flush()
    return entry


# --- reading ----------------------------------------------------------------------------------

# A sentence ends at . ! or ? followed by a space and a capital letter, quote or bracket.
# A digit does not start a sentence here: "Fig. 3" and "Eq. 2" are not boundaries.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(\[])")


def first_sentence(text: str, max_chars: int) -> str:
    """The text's first sentence, cut at `max_chars` (at a space, with an ellipsis)."""
    flat = " ".join(text.split())
    sentence = _SENTENCE_END.split(flat, maxsplit=1)[0]
    if len(sentence) <= max_chars:
        return sentence
    cut = sentence[: max_chars - 1]
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut.rstrip(" ,;:.-") + "…"


def card_summary(tldr: str | None, abstract: str) -> tuple[str, Literal["analysis", "abstract"]]:
    """The card's one-liner: the stored analysis TL;DR if there is one, else the
    abstract's first sentence. No LLM call."""
    if tldr and tldr.strip():
        return " ".join(tldr.split()), "analysis"
    return first_sentence(abstract, get_settings().digest_summary_max_chars), "abstract"


@dataclass(frozen=True)
class DigestCard:
    entry: DigestEntry
    paper: Paper
    best_interest_name: str | None
    summary: str
    summary_source: Literal["analysis", "abstract"]
    trending: bool


@dataclass(frozen=True)
class DigestView:
    requested_date: date
    digest_date: date | None  # None: no digest exists yet
    cards: list[DigestCard]
    recent_dates: list[date]  # newest first
    has_interests: bool  # at least one active interest


RECENT_DATES = 14


def _cards(session: Session, digest_date: date, arxiv_id: str | None = None) -> list[DigestCard]:
    """The cards of one digest in rank order (or just one paper's card)."""
    query = (
        select(DigestEntry, Paper, Interest.name, PaperAnalysis.tldr)
        .join(Paper, Paper.arxiv_id == DigestEntry.arxiv_id)
        .outerjoin(Interest, Interest.id == DigestEntry.best_interest_id)
        .outerjoin(PaperAnalysis, PaperAnalysis.arxiv_id == DigestEntry.arxiv_id)
        .where(DigestEntry.digest_date == digest_date)
        .order_by(DigestEntry.rank, DigestEntry.arxiv_id)
    )
    if arxiv_id is not None:
        query = query.where(DigestEntry.arxiv_id == arxiv_id)
    threshold = get_settings().digest_trending_upvotes
    cards = []
    for entry, paper, interest_name, tldr in session.execute(query).all():
        summary, source = card_summary(tldr, paper.abstract)
        cards.append(
            DigestCard(
                entry=entry,
                paper=paper,
                best_interest_name=interest_name,
                summary=summary,
                summary_source=source,
                trending=paper.hf_upvotes is not None and paper.hf_upvotes >= threshold,
            )  # fmt: skip
        )
    return cards


def load_card(session: Session, digest_date: date, arxiv_id: str) -> DigestCard | None:
    cards = _cards(session, digest_date, arxiv_id)
    return cards[0] if cards else None


def load_digest(
    session: Session, requested: date | None = None, now: datetime | None = None
) -> DigestView:
    """The digest for `requested` (default today), or the latest one at or before it, or
    the newest of all. Read-only: it never builds a digest."""
    requested = requested or today_in_zone(now)
    dates = list(
        session.scalars(
            select(DigestEntry.digest_date)
            .distinct()
            .order_by(DigestEntry.digest_date.desc())
            .limit(500)
        )
    )
    chosen = next((d for d in dates if d <= requested), dates[0] if dates else None)
    has_interests = (
        session.scalar(select(Interest.id).where(Interest.is_active.is_(True)).limit(1)) is not None
    )
    cards = _cards(session, chosen) if chosen is not None else []
    return DigestView(requested, chosen, cards, dates[:RECENT_DATES], has_interests)

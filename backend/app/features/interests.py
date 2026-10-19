"""Interests: create, read, update and delete, and the text that gets embedded.

No function here calls an API. An interest's embedding is made later by the scoring task
(app/features/digest.py), which compares the hash stored next to the vector with the hash
of the interest's current text: editing the name, description or keywords changes the
hash, so the next scoring run embeds it again. Editing only categories, `is_active` or
the weight does not.
"""

import hashlib
from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.models import Interest


class DuplicateInterestError(ValueError):
    """An interest with this name exists already."""


def normalize_keywords(keywords: Sequence[str]) -> list[str]:
    """Trimmed, no blanks, no repeats (case-insensitive; the first spelling is kept)."""
    seen: set[str] = set()
    result: list[str] = []
    for keyword in keywords:
        cleaned = " ".join(keyword.split())
        if cleaned and cleaned.casefold() not in seen:
            seen.add(cleaned.casefold())
            result.append(cleaned)
    return result


def embedding_text(name: str, description: str | None, keywords: Sequence[str]) -> str:
    """What gets embedded for an interest: its name, description and keywords."""
    parts = [name.strip()]
    if description and description.strip():
        parts.append(description.strip())
    if keywords:
        parts.append("Keywords: " + ", ".join(keywords))
    return "\n\n".join(parts)


def text_hash(model: str, text: str) -> str:
    """Identifies one embedding: the model and the exact text. Changing either means the
    stored vector is stale."""
    return hashlib.sha256(f"{model}\n{text}".encode()).hexdigest()


def current_hash(interest: Interest) -> str:
    return text_hash(
        get_settings().embedding_model,
        embedding_text(interest.name, interest.description, interest.keywords),
    )


def has_current_embedding(interest: Interest) -> bool:
    return interest.embedding is not None and interest.embedding_hash == current_hash(interest)


def list_interests(session: Session) -> list[Interest]:
    return list(session.scalars(select(Interest).order_by(Interest.name, Interest.id)))


def get_interest(session: Session, interest_id: int) -> Interest | None:
    return session.get(Interest, interest_id)


def create_interest(
    session: Session,
    *,
    name: str,
    description: str | None,
    keywords: Sequence[str],
    categories: Sequence[str],
    is_active: bool = True,
) -> Interest:
    if session.scalar(select(Interest.id).where(Interest.name == name)) is not None:
        raise DuplicateInterestError(name)
    interest = Interest(
        name=name,
        description=description,
        keywords=normalize_keywords(keywords),
        categories=list(dict.fromkeys(categories)),
        is_active=is_active,
        weight=1.0,
    )
    session.add(interest)
    try:
        session.flush()
    except IntegrityError as err:  # a concurrent insert of the same name
        session.rollback()
        raise DuplicateInterestError(name) from err
    session.refresh(interest)  # server defaults (timestamps)
    return interest


def update_interest(
    session: Session,
    interest_id: int,
    *,
    name: str,
    description: str | None,
    keywords: Sequence[str],
    categories: Sequence[str],
    is_active: bool,
) -> Interest | None:
    """Replace the editable fields (the weight is only changed by Save and Dismiss).
    None when the interest does not exist."""
    interest = session.get(Interest, interest_id)
    if interest is None:
        return None
    clash = session.scalar(
        select(Interest.id).where(Interest.name == name, Interest.id != interest_id)
    )
    if clash is not None:
        raise DuplicateInterestError(name)
    interest.name = name
    interest.description = description
    interest.keywords = normalize_keywords(keywords)
    interest.categories = list(dict.fromkeys(categories))
    interest.is_active = is_active
    session.flush()
    session.refresh(interest)
    return interest


def delete_interest(session: Session, interest_id: int) -> bool:
    interest = session.get(Interest, interest_id)
    if interest is None:
        return False
    session.delete(interest)  # digest_entries.best_interest_id becomes NULL (FK SET NULL)
    session.flush()
    return True

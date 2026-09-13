from __future__ import annotations

from collections.abc import Iterable

from app.api.errors import ValidationError


def validate_user_id(raw_value: str | None, known_user_ids: Iterable[int]) -> int:
    """Input Validation Entity (spec 13.5): user_id must be a whole number
    present in the dataset."""
    if raw_value is None or not str(raw_value).strip():
        raise ValidationError("Please enter a user ID.")
    try:
        user_id = int(str(raw_value).strip())
    except ValueError:
        raise ValidationError("User ID must be a whole number.") from None
    if user_id not in set(known_user_ids):
        raise ValidationError(f"User ID {user_id} is not in the dataset.")
    return user_id


def validate_movie_title(raw_value: str | None, known_titles: Iterable[str]) -> str:
    """Input Validation Entity (spec 13.5): movie_title must exist in the
    filtered recommendable set."""
    if raw_value is None or not str(raw_value).strip():
        raise ValidationError("Please enter a movie title.")
    title = str(raw_value).strip()
    if title not in set(known_titles):
        raise ValidationError(f'"{title}" is not in the recommendable movie set.')
    return title

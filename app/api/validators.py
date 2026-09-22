from __future__ import annotations

from collections.abc import Iterable

from app.api.errors import ValidationError


def validate_user_id(raw_value: str | None, known_user_ids: Iterable[int]) -> int:
    """Input Validation Entity (spec 13.5): user_id must be a whole number
    present in the dataset. The upper bound named in the error message is
    read from known_user_ids itself, not a literal, so it always matches
    the loaded model."""
    if raw_value is None or not str(raw_value).strip():
        raise ValidationError("Please enter a viewer ID.")
    try:
        user_id = int(str(raw_value).strip())
    except ValueError:
        raise ValidationError("Viewer ID must be a number.") from None
    known = set(known_user_ids)
    if user_id not in known:
        raise ValidationError(f"There is no viewer {user_id}. Viewer IDs run from 1 to {max(known)}.")
    return user_id


def validate_movie_title(raw_value: str | None, known_titles: Iterable[str]) -> str:
    """Input Validation Entity (spec 13.5): movie_title must exist in the
    filtered recommendable set."""
    if raw_value is None or not str(raw_value).strip():
        raise ValidationError("Please enter a film title.")
    title = str(raw_value).strip()
    if title not in set(known_titles):
        raise ValidationError(f'We couldn\'t find "{title}". Pick a title from the suggestions as you type.')
    return title


def validate_genre(raw_value: str | None, known_genres: Iterable[str]) -> str:
    """Input Validation Entity (spec 13.5): genre must be one of the genres
    present in the recommendable movie catalog."""
    if raw_value is None or not str(raw_value).strip():
        raise ValidationError("Please choose a genre.")
    genre = str(raw_value).strip()
    if genre not in set(known_genres):
        raise ValidationError(f'"{genre}" is not a genre in this dataset.')
    return genre

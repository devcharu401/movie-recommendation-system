from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pandas as pd

ENCODING = "ISO-8859-1"

GENRE_COLUMNS = [
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]

USER_COLUMNS = ["user_id", "age", "gender", "occupation", "zip_code"]

RATING_COLUMNS = ["user_id", "movie_id", "rating", "timestamp"]

ITEM_COLUMNS = [
    "movie_id", "movie_title", "release_date", "video_release_date", "imdb_url",
] + GENRE_COLUMNS


def load_user_data(path: Path) -> pd.DataFrame:
    """User Data Loader (spec 13.1): reads u.user into the users schema shape."""
    return pd.read_csv(path, sep="|", names=USER_COLUMNS, header=None, encoding=ENCODING)


def load_movie_data(path: Path) -> pd.DataFrame:
    """Movie Data Loader (spec 13.1): reads u.item, collapsing the 19 genre flags
    into a single pipe-delimited genre string to fit genre VARCHAR(100)."""
    df = pd.read_csv(path, sep="|", names=ITEM_COLUMNS, header=None, encoding=ENCODING)
    df["release_date"] = df["release_date"].apply(_parse_release_date)
    df["genre"] = df[GENRE_COLUMNS].apply(_collapse_genres, axis=1)
    return df[["movie_id", "movie_title", "release_date", "genre"]]


def load_ratings_data(path: Path) -> pd.DataFrame:
    """Ratings Data Loader (spec 13.1): reads u.data. MovieLens carries no rating
    identifier; rating_id is generated as a surrogate key at insert time."""
    df = pd.read_csv(path, sep="\t", names=RATING_COLUMNS, header=None, encoding=ENCODING)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    return df


def validate_data_format(users: pd.DataFrame, movies: pd.DataFrame, ratings: pd.DataFrame) -> None:
    """Data Format Validator (spec 13.1): checks column structure, value ranges,
    and referential consistency across the three loaded datasets."""
    _require_columns(users, USER_COLUMNS)
    _require_columns(movies, ["movie_id", "movie_title", "release_date", "genre"])
    _require_columns(ratings, RATING_COLUMNS)

    if users["user_id"].duplicated().any():
        raise ValueError("duplicate user_id values in u.user")
    if movies["movie_id"].duplicated().any():
        raise ValueError("duplicate movie_id values in u.item")
    if not ratings["rating"].between(1, 5).all():
        raise ValueError("rating values outside the expected 1-5 range")

    unknown_users = set(ratings["user_id"]) - set(users["user_id"])
    if unknown_users:
        raise ValueError(f"ratings reference unknown user_id values: {sorted(unknown_users)[:5]}")

    unknown_movies = set(ratings["movie_id"]) - set(movies["movie_id"])
    if unknown_movies:
        raise ValueError(f"ratings reference unknown movie_id values: {sorted(unknown_movies)[:5]}")


def _parse_release_date(value: object) -> date | None:
    if pd.isna(value) or not str(value).strip():
        return None
    return datetime.strptime(str(value), "%d-%b-%Y").date()


def _collapse_genres(row: pd.Series) -> str:
    genres = [name for name in GENRE_COLUMNS if name != "unknown" and row[name] == 1]
    return "|".join(genres)


def _require_columns(df: pd.DataFrame, expected: list[str]) -> None:
    missing = set(expected) - set(df.columns)
    if missing:
        raise ValueError(f"missing expected columns: {sorted(missing)}")

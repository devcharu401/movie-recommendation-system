from __future__ import annotations

import pandas as pd

from app.repositories.movie_repository import get_all_movies
from app.repositories.rating_repository import get_all_ratings
from app.repositories.user_repository import get_all_users


def handle_missing_values(
    users: pd.DataFrame, movies: pd.DataFrame, ratings: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Missing Value Handler (spec 13.2). Movie release_date is the one field
    with legitimate blanks (spec 3.4) and is left as-is; rows missing an
    identifying or rating field are dropped since the matrix build downstream
    cannot use them."""
    users = users.dropna(subset=["user_id", "age", "gender", "occupation"])
    movies = movies.copy()
    movies["genre"] = movies["genre"].fillna("")
    movies = movies.dropna(subset=["movie_id", "movie_title"])
    ratings = ratings.dropna(subset=["user_id", "movie_id", "rating"])
    return users, movies, ratings


def remove_duplicate_records(ratings: pd.DataFrame) -> pd.DataFrame:
    """Duplicate Record Remover (spec 13.2): a user rates a given movie once;
    if the same pair appears twice, keep the later rating."""
    return (
        ratings.sort_values("timestamp")
        .drop_duplicates(subset=["user_id", "movie_id"], keep="last")
        .reset_index(drop=True)
    )


def filter_by_minimum_ratings(ratings: pd.DataFrame, min_ratings: int) -> pd.DataFrame:
    """Data Filtering Entity (spec 13.2): the synopsis's candidate filter —
    only movies with at least min_ratings ratings are recommendable."""
    counts = ratings.groupby("movie_id")["rating"].transform("count")
    return ratings[counts >= min_ratings].reset_index(drop=True)


def merge_datasets(users: pd.DataFrame, movies: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    """Dataset Merger (spec 13.2): joins the three tables into one analysis
    frame, restricted to the movies that survived filtering."""
    merged = ratings.merge(users, on="user_id", how="inner")
    merged = merged.merge(movies, on="movie_id", how="inner")
    return merged


def build_preprocessed_dataset(min_ratings: int) -> pd.DataFrame:
    """Runs the full Data Preprocessing Module (spec 13.2) against the
    database, via the repositories, and returns the merged analysis frame."""
    users = pd.DataFrame(get_all_users())
    movies = pd.DataFrame(get_all_movies())
    ratings = pd.DataFrame(get_all_ratings())

    users, movies, ratings = handle_missing_values(users, movies, ratings)
    ratings = remove_duplicate_records(ratings)
    ratings = filter_by_minimum_ratings(ratings, min_ratings)

    return merge_datasets(users, movies, ratings)

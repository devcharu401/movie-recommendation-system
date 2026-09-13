from __future__ import annotations

import pandas as pd
from scipy.sparse import csr_matrix


def build_user_movie_matrix(merged: pd.DataFrame) -> pd.DataFrame:
    """User-Movie Matrix Generator (spec 13.3): pivots the merged, filtered
    ratings into a dense user-by-movie matrix, one row per user, one column
    per movie title, unrated cells filled with 0."""
    labels = _disambiguate_titles(merged)
    return merged.assign(_movie_label=labels).pivot_table(
        index="user_id", columns="_movie_label", values="rating", fill_value=0
    )


def _disambiguate_titles(merged: pd.DataFrame) -> pd.Series:
    """MovieLens 100K carries at least one duplicate title under two distinct
    movie_ids ('Chasing Amy (1997)' is both 246 and 268); pivoting on the raw
    title would silently merge them into a single, averaged column. Suffix
    the movie_id onto any non-unique title so every movie keeps its own
    column."""
    id_title = merged[["movie_id", "movie_title"]].drop_duplicates()
    duplicate_titles = set(id_title.loc[id_title["movie_title"].duplicated(keep=False), "movie_title"])
    if not duplicate_titles:
        return merged["movie_title"]
    is_duplicate = merged["movie_title"].isin(duplicate_titles)
    suffixed = merged["movie_title"] + " (#" + merged["movie_id"].astype(str) + ")"
    return suffixed.where(is_duplicate, merged["movie_title"])


def build_user_features(user_movie_matrix: pd.DataFrame) -> pd.DataFrame:
    """Feature Vector Builder, user orientation (spec 13.3): each row is one
    user's rating vector across all qualifying movies — user_feature_df on
    the class diagram (spec 11.5)."""
    return user_movie_matrix


def build_movie_features(user_movie_matrix: pd.DataFrame) -> pd.DataFrame:
    """Feature Vector Builder, movie orientation (spec 13.3): each row is one
    movie's rating vector across all users — movie_feature_df on the class
    diagram (spec 11.5)."""
    return user_movie_matrix.T


def build_movie_catalog(merged: pd.DataFrame) -> pd.DataFrame:
    """Maps each disambiguated movie_label used as a matrix row/column back to
    its movie_id, title, genre and release_date, so the recommendation engine
    can assemble result records from a label alone."""
    labels = _disambiguate_titles(merged)
    catalog = merged.assign(movie_label=labels)[
        ["movie_label", "movie_id", "movie_title", "genre", "release_date"]
    ].drop_duplicates(subset="movie_label")
    return catalog.set_index("movie_label")


def to_sparse_matrix(feature_df: pd.DataFrame) -> csr_matrix:
    """Sparse Matrix Handler (spec 13.3): the ratings matrix is overwhelmingly
    zeros, so the KNN engine works against a csr_matrix rather than the dense
    DataFrame it's built from."""
    return csr_matrix(feature_df.values)

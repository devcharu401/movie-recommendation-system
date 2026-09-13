from __future__ import annotations

import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

HIGH_RATING_THRESHOLD = 4


def fit_nearest_neighbors(feature_matrix: csr_matrix) -> NearestNeighbors:
    """KNN Neighbor Selector (spec 13.4): brute-force cosine-metric neighbor
    search, per synopsis §2 (locked algorithm choice)."""
    model = NearestNeighbors(metric="cosine", algorithm="brute")
    model.fit(feature_matrix)
    return model


def similarity_from_distance(distance: float) -> float:
    """Similarity Calculator (spec 13.4): sklearn's cosine metric returns
    1 - cosine similarity as distance; invert it back to a similarity score."""
    return 1.0 - distance


def rank_candidates(scores: dict[str, float], top_n: int) -> list[tuple[str, float]]:
    """Recommendation Ranking Engine (spec 13.4): highest score first, capped
    at top_n. top_n and every other count come from config at the call site,
    never hardcoded here."""
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_n]


def recommend_user_based(
    user_id: int,
    user_feature_df: pd.DataFrame,
    model: NearestNeighbors,
    movie_catalog: pd.DataFrame,
    similar_users_count: int,
    top_n: int,
) -> list[dict]:
    """User-Based Recommendation Entity (spec 13.4): finds the
    similar_users_count nearest users by rating pattern, collects the movies
    they rated highly, excludes anything the target user already rated, and
    returns the top_n ranked candidates."""
    if user_id not in user_feature_df.index:
        raise ValueError(f"unknown user_id: {user_id}")

    target_vector = user_feature_df.loc[[user_id]].values
    distances, indices = model.kneighbors(target_vector, n_neighbors=similar_users_count + 1)

    neighbors = [
        (user_feature_df.index[position], similarity_from_distance(distance))
        for position, distance in zip(indices[0], distances[0])
        if user_feature_df.index[position] != user_id
    ][:similar_users_count]

    already_rated = set(user_feature_df.columns[user_feature_df.loc[user_id] > 0])

    scores: dict[str, float] = {}
    for neighbor_id, similarity in neighbors:
        neighbor_ratings = user_feature_df.loc[neighbor_id]
        highly_rated = neighbor_ratings[neighbor_ratings >= HIGH_RATING_THRESHOLD]
        for movie_label, rating in highly_rated.items():
            if movie_label in already_rated:
                continue
            scores[movie_label] = scores.get(movie_label, 0.0) + similarity * rating

    ranked = rank_candidates(scores, top_n)
    return _build_result_records(ranked, movie_catalog)


def recommend_item_based(
    movie_title: str,
    movie_feature_df: pd.DataFrame,
    model: NearestNeighbors,
    movie_catalog: pd.DataFrame,
    top_n: int,
) -> list[dict]:
    """Item-Based Recommendation Entity (spec 13.4): finds the top_n nearest
    movies to the given title by rating pattern across users, excluding the
    query movie itself."""
    label = _resolve_movie_label(movie_title, movie_catalog)

    target_vector = movie_feature_df.loc[[label]].values
    distances, indices = model.kneighbors(target_vector, n_neighbors=top_n + 1)

    candidates = {
        movie_feature_df.index[position]: similarity_from_distance(distance)
        for position, distance in zip(indices[0], distances[0])
        if movie_feature_df.index[position] != label
    }

    ranked = rank_candidates(candidates, top_n)
    return _build_result_records(ranked, movie_catalog)


def _resolve_movie_label(movie_title: str, movie_catalog: pd.DataFrame) -> str:
    if movie_title in movie_catalog.index:
        return movie_title
    matches = movie_catalog.index[movie_catalog["movie_title"] == movie_title]
    if len(matches) == 0:
        raise ValueError(f"unknown movie title: {movie_title}")
    if len(matches) > 1:
        raise ValueError(f"ambiguous movie title {movie_title!r}; matches: {list(matches)}")
    return matches[0]


def _build_result_records(ranked: list[tuple[str, float]], movie_catalog: pd.DataFrame) -> list[dict]:
    records = []
    for label, score in ranked:
        row = movie_catalog.loc[label]
        records.append(
            {
                "movie_id": int(row["movie_id"]),
                "movie_title": row["movie_title"],
                "genre": row["genre"],
                "release_date": row["release_date"],
                "score": float(score),
            }
        )
    return records

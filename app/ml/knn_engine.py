from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

HIGH_RATING_THRESHOLD = 4
NEIGHBOR_TOP_MOVIES_CAP = 5


@dataclass(frozen=True)
class Neighbor:
    """One of the k nearest users behind a user-based recommendation set
    (spec 13.4). similarity is the cosine similarity to the target user,
    bounded 0-1; top_rated_movies is this neighbour's own highly rated
    titles, excluding anything the target user already rated, for a later
    'what else they liked' panel."""

    user_id: int
    similarity: float
    top_rated_movies: tuple[str, ...]


@dataclass(frozen=True)
class NeighborRating:
    """A single neighbour's rating of a recommended movie."""

    neighbor_id: int
    rating: int


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
) -> tuple[list[dict], list[Neighbor], dict[int, list[NeighborRating]]]:
    """User-Based Recommendation Entity (spec 13.4): finds the
    similar_users_count nearest users by rating pattern, collects the movies
    they rated highly, excludes anything the target user already rated, and
    returns the top_n ranked candidates.

    Also returns the neighbours behind that ranking (KNN Neighbor Selector
    evidence, spec 13.4) and, per recommended movie, which of those
    neighbours rated it and with what score — read from the in-memory
    user-movie matrix, with no extra queries or recomputed similarities."""
    if user_id not in user_feature_df.index:
        raise ValueError(f"unknown user_id: {user_id}")

    target_vector = user_feature_df.loc[[user_id]].values
    distances, indices = model.kneighbors(target_vector, n_neighbors=similar_users_count + 1)

    neighbor_pairs = [
        (user_feature_df.index[position], similarity_from_distance(distance))
        for position, distance in zip(indices[0], distances[0])
        if user_feature_df.index[position] != user_id
    ][:similar_users_count]

    already_rated = set(user_feature_df.columns[user_feature_df.loc[user_id] > 0])

    scores: dict[str, float] = {}
    neighbor_similarity: dict[str, float] = {}
    for neighbor_id, similarity in neighbor_pairs:
        neighbor_ratings = user_feature_df.loc[neighbor_id]
        highly_rated = neighbor_ratings[neighbor_ratings >= HIGH_RATING_THRESHOLD]
        for movie_label, rating in highly_rated.items():
            if movie_label in already_rated:
                continue
            scores[movie_label] = scores.get(movie_label, 0.0) + similarity * rating
            # neighbor_pairs is ordered by descending similarity, so the first
            # neighbor to surface a movie is its most similar contributor.
            neighbor_similarity.setdefault(movie_label, similarity)

    ranked = rank_candidates(scores, top_n)
    recommendations = _build_user_based_records(ranked, movie_catalog, neighbor_similarity)

    neighbors = [
        Neighbor(
            user_id=int(neighbor_id),
            similarity=float(similarity),
            top_rated_movies=_neighbor_top_rated_titles(neighbor_id, user_feature_df, movie_catalog, already_rated),
        )
        for neighbor_id, similarity in neighbor_pairs
    ]

    neighbor_ratings = {
        int(movie_catalog.loc[label, "movie_id"]): _neighbors_who_rated(label, neighbor_pairs, user_feature_df)
        for label, _ in ranked
    }

    return recommendations, neighbors, neighbor_ratings


def _neighbor_top_rated_titles(
    neighbor_id: object,
    user_feature_df: pd.DataFrame,
    movie_catalog: pd.DataFrame,
    already_rated: set[str],
) -> tuple[str, ...]:
    neighbor_ratings = user_feature_df.loc[neighbor_id]
    candidates = neighbor_ratings[(neighbor_ratings > 0) & ~neighbor_ratings.index.isin(already_rated)]

    # Many films tie at the neighbour's own maximum rating. Break the tie by
    # rating_count (how widely the film is rated overall, precomputed on
    # movie_catalog at model-build time) rather than leaving it to sort
    # order, which would otherwise default to the pivot table's alphabetical
    # column order. Sorting on two columns routes through np.lexsort, which
    # is stable, so this is deterministic without naming a sort kind.
    ranking = pd.DataFrame(
        {"rating": candidates, "rating_count": movie_catalog.loc[candidates.index, "rating_count"]}
    ).sort_values(by=["rating", "rating_count"], ascending=False)

    top_labels = ranking.index[:NEIGHBOR_TOP_MOVIES_CAP]
    return tuple(movie_catalog.loc[label, "movie_title"] for label in top_labels)


def _neighbors_who_rated(
    movie_label: str,
    neighbor_pairs: list[tuple[object, float]],
    user_feature_df: pd.DataFrame,
) -> list[NeighborRating]:
    column = user_feature_df[movie_label]
    return [
        NeighborRating(neighbor_id=int(neighbor_id), rating=int(column.loc[neighbor_id]))
        for neighbor_id, _ in neighbor_pairs
        if column.loc[neighbor_id] > 0
    ]


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
    return _build_item_based_records(ranked, movie_catalog)


def _resolve_movie_label(movie_title: str, movie_catalog: pd.DataFrame) -> str:
    if movie_title in movie_catalog.index:
        return movie_title
    matches = movie_catalog.index[movie_catalog["movie_title"] == movie_title]
    if len(matches) == 0:
        raise ValueError(f"unknown movie title: {movie_title}")
    if len(matches) > 1:
        raise ValueError(f"ambiguous movie title {movie_title!r}; matches: {list(matches)}")
    return matches[0]


def _base_record(label: str, movie_catalog: pd.DataFrame) -> dict:
    row = movie_catalog.loc[label]
    return {
        "movie_id": int(row["movie_id"]),
        "movie_title": row["movie_title"],
        "genre": row["genre"],
        "release_date": row["release_date"],
    }


def _build_item_based_records(ranked: list[tuple[str, float]], movie_catalog: pd.DataFrame) -> list[dict]:
    records = []
    for label, similarity in ranked:
        record = _base_record(label, movie_catalog)
        record["similarity"] = float(similarity)
        records.append(record)
    return records


def _build_user_based_records(
    ranked: list[tuple[str, float]], movie_catalog: pd.DataFrame, neighbor_similarity: dict[str, float]
) -> list[dict]:
    records = []
    for label, rank_score in ranked:
        record = _base_record(label, movie_catalog)
        record["rank_score"] = float(rank_score)
        record["similarity"] = float(neighbor_similarity[label])
        records.append(record)
    return records

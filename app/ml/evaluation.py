from __future__ import annotations

import random
from dataclasses import dataclass
from statistics import mean

import pandas as pd
from sklearn.model_selection import train_test_split

from app.ml.feature_builder import build_movie_catalog, build_user_features, build_user_movie_matrix, to_sparse_matrix
from app.ml.knn_engine import HIGH_RATING_THRESHOLD, fit_nearest_neighbors, recommend_user_based

RANDOM_SEED = 42
DEFAULT_TEST_FRACTION = 0.2
DEFAULT_TOP_N = 10
DEFAULT_SAMPLE_SIZE = 200
DEFAULT_NEIGHBOR_COUNTS = [3, 5, 10, 15, 20]


@dataclass
class KSensitivityResult:
    neighbor_count: int
    precision_at_k: float
    recall_at_k: float
    coverage: float
    evaluated_users: int


@dataclass
class EvaluationReport:
    matrix_shape: tuple[int, int]
    sparsity: float
    train_rows: int
    test_rows: int
    sample_size: int
    top_n: int
    results: list[KSensitivityResult]


def split_ratings(
    merged: pd.DataFrame, test_fraction: float = DEFAULT_TEST_FRACTION, seed: int = RANDOM_SEED
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Held-out split (spec 10): a reproducible random split of the merged,
    filtered ratings into train and test rows."""
    return train_test_split(merged, test_size=test_fraction, random_state=seed)


def compute_sparsity(user_movie_matrix: pd.DataFrame) -> float:
    """Sparsity (spec 10): share of the user-movie matrix that is unrated."""
    total_cells = user_movie_matrix.shape[0] * user_movie_matrix.shape[1]
    rated_cells = int((user_movie_matrix.values != 0).sum())
    return 1.0 - (rated_cells / total_cells)


def evaluate_user_based(
    train_merged: pd.DataFrame,
    test_ratings: pd.DataFrame,
    neighbor_count: int,
    top_n: int,
    sample_user_ids: list[int],
) -> KSensitivityResult:
    """Precision@top_n, Recall@top_n and catalogue coverage (spec 10) for the
    user-based path at a given neighbour count. "Relevant" means the user
    rated the movie at least HIGH_RATING_THRESHOLD in the held-out test
    rows — the same threshold the recommender itself uses for "rated
    highly", so the metric and the algorithm agree on what counts."""
    user_movie_matrix = build_user_movie_matrix(train_merged)
    user_feature_df = build_user_features(user_movie_matrix)
    movie_catalog = build_movie_catalog(train_merged)
    model = fit_nearest_neighbors(to_sparse_matrix(user_feature_df))

    precisions: list[float] = []
    recalls: list[float] = []
    recommended_movie_ids: set[int] = set()

    for user_id in sample_user_ids:
        if user_id not in user_feature_df.index:
            continue

        recommendations = recommend_user_based(
            user_id, user_feature_df, model, movie_catalog, neighbor_count, top_n
        )
        recommended_ids = {r["movie_id"] for r in recommendations}
        recommended_movie_ids |= recommended_ids

        relevant_ids = set(
            test_ratings.loc[
                (test_ratings["user_id"] == user_id) & (test_ratings["rating"] >= HIGH_RATING_THRESHOLD),
                "movie_id",
            ]
        )
        if not relevant_ids:
            continue

        hits = len(recommended_ids & relevant_ids)
        precisions.append(hits / top_n)
        recalls.append(hits / len(relevant_ids))

    catalogue_size = movie_catalog["movie_id"].nunique()
    coverage = len(recommended_movie_ids) / catalogue_size if catalogue_size else 0.0

    return KSensitivityResult(
        neighbor_count=neighbor_count,
        precision_at_k=mean(precisions) if precisions else 0.0,
        recall_at_k=mean(recalls) if recalls else 0.0,
        coverage=coverage,
        evaluated_users=len(precisions),
    )


def run_evaluation(
    merged: pd.DataFrame,
    neighbor_counts: list[int] = DEFAULT_NEIGHBOR_COUNTS,
    top_n: int = DEFAULT_TOP_N,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    test_fraction: float = DEFAULT_TEST_FRACTION,
    seed: int = RANDOM_SEED,
) -> EvaluationReport:
    """Runs the full evaluation suite (spec 10): sparsity of the full matrix,
    then Precision@top_n / Recall@top_n / coverage for each neighbour count
    in neighbor_counts (the k sensitivity table), over a reproducible sample
    of training users."""
    train_merged, test_merged = split_ratings(merged, test_fraction, seed)

    full_matrix = build_user_movie_matrix(merged)
    sparsity = compute_sparsity(full_matrix)

    train_user_ids = sorted(train_merged["user_id"].unique().tolist())
    sample_user_ids = (
        sorted(random.Random(seed).sample(train_user_ids, sample_size))
        if sample_size < len(train_user_ids)
        else train_user_ids
    )

    results = [
        evaluate_user_based(train_merged, test_merged, k, top_n, sample_user_ids) for k in neighbor_counts
    ]

    return EvaluationReport(
        matrix_shape=full_matrix.shape,
        sparsity=sparsity,
        train_rows=len(train_merged),
        test_rows=len(test_merged),
        sample_size=len(sample_user_ids),
        top_n=top_n,
        results=results,
    )

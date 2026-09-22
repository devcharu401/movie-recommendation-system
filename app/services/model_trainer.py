from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from app.ml.evaluation import EvaluationReport
from app.ml.feature_builder import (
    build_movie_catalog,
    build_movie_features,
    build_user_features,
    build_user_movie_matrix,
    to_sparse_matrix,
)
from app.ml.knn_engine import fit_nearest_neighbors
from app.ml.preprocessing import build_preprocessed_dataset

logger = logging.getLogger(__name__)

ARTIFACT_FILES = {
    "user_feature_df": "user_feature_df.joblib",
    "movie_feature_df": "movie_feature_df.joblib",
    "movie_catalog": "movie_catalog.joblib",
    "merged_dataset": "merged_dataset.joblib",
    "user_model": "user_model.joblib",
    "movie_model": "movie_model.joblib",
}

EVALUATION_REPORT_FILE = "evaluation_report.joblib"


@dataclass
class ModelArtifacts:
    user_feature_df: pd.DataFrame
    movie_feature_df: pd.DataFrame
    movie_catalog: pd.DataFrame
    merged_dataset: pd.DataFrame
    user_model: NearestNeighbors
    movie_model: NearestNeighbors


class ModelTrainer:
    """Model Integration lifecycle (spec 13.6): owns the train/persist/load
    of model artifacts, so app/services/recommendation_service.py never
    touches the filesystem or the ML pipeline directly."""

    def __init__(self, models_dir: Path, min_ratings_per_movie: int) -> None:
        self._models_dir = models_dir
        self._min_ratings_per_movie = min_ratings_per_movie

    def load(self) -> ModelArtifacts:
        missing = [name for name, filename in ARTIFACT_FILES.items() if not (self._models_dir / filename).exists()]
        if missing:
            raise FileNotFoundError(
                f"missing model artifacts {missing} in {self._models_dir}; run scripts/train_model.py first"
            )
        loaded = {name: joblib.load(self._models_dir / filename) for name, filename in ARTIFACT_FILES.items()}
        return ModelArtifacts(**loaded)

    def train(self) -> ModelArtifacts:
        merged = build_preprocessed_dataset(self._min_ratings_per_movie)

        user_movie_matrix = build_user_movie_matrix(merged)
        user_feature_df = build_user_features(user_movie_matrix)
        movie_feature_df = build_movie_features(user_movie_matrix)
        movie_catalog = build_movie_catalog(merged)

        user_model = fit_nearest_neighbors(to_sparse_matrix(user_feature_df))
        movie_model = fit_nearest_neighbors(to_sparse_matrix(movie_feature_df))

        artifacts = ModelArtifacts(
            user_feature_df=user_feature_df,
            movie_feature_df=movie_feature_df,
            movie_catalog=movie_catalog,
            merged_dataset=merged,
            user_model=user_model,
            movie_model=movie_model,
        )
        self._persist(artifacts)
        return artifacts

    def _persist(self, artifacts: ModelArtifacts) -> None:
        self._models_dir.mkdir(parents=True, exist_ok=True)
        for name, filename in ARTIFACT_FILES.items():
            path = self._models_dir / filename
            joblib.dump(getattr(artifacts, name), path)
            logger.info("wrote %s", path)

    def persist_evaluation(self, report: EvaluationReport) -> None:
        """Stores the evaluation report (spec 10) alongside the model
        artifacts. Kept separate from ARTIFACT_FILES/load(): the report is
        optional evidence for the About page, not something recommend()
        depends on, so a missing report must not fail app startup."""
        self._models_dir.mkdir(parents=True, exist_ok=True)
        path = self._models_dir / EVALUATION_REPORT_FILE
        joblib.dump(report, path)
        logger.info("wrote %s", path)

    def load_evaluation(self) -> EvaluationReport | None:
        path = self._models_dir / EVALUATION_REPORT_FILE
        if not path.exists():
            return None
        return joblib.load(path)

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.neighbors import NearestNeighbors

from app.ml.knn_engine import recommend_item_based, recommend_user_based
from app.services.model_trainer import ModelArtifacts, ModelTrainer
from app.services.visualization_service import generate_all_figures


class Recommendation:
    """Recommendation (class diagram, spec 11.5): holds the trained feature
    matrices and KNN models for the process lifetime. Artifacts are loaded
    once here, at construction, not per recommend() call."""

    def __init__(
        self,
        trainer: ModelTrainer,
        similar_users_count: int,
        top_n_recommendations: int,
        figures_dir: Path,
    ) -> None:
        self._trainer = trainer
        self._similar_users_count = similar_users_count
        self._top_n_recommendations = top_n_recommendations
        self._figures_dir = figures_dir

        self.user_feature_df: pd.DataFrame
        self.movie_feature_df: pd.DataFrame
        self._movie_catalog: pd.DataFrame
        self._merged_dataset: pd.DataFrame
        self._user_model: NearestNeighbors
        self._movie_model: NearestNeighbors

        self._apply(self._load_artifacts())

    def _load_artifacts(self) -> ModelArtifacts:
        try:
            return self._trainer.load()
        except FileNotFoundError as exc:
            raise RuntimeError(
                "no trained model artifacts found; run scripts/train_model.py before starting the app"
            ) from exc

    def _apply(self, artifacts: ModelArtifacts) -> None:
        self.user_feature_df = artifacts.user_feature_df
        self.movie_feature_df = artifacts.movie_feature_df
        self._movie_catalog = artifacts.movie_catalog
        self._merged_dataset = artifacts.merged_dataset
        self._user_model = artifacts.user_model
        self._movie_model = artifacts.movie_model

    def train_model(self) -> None:
        """Rebuilds and persists the matrices and models, then reloads this
        instance's state from the new artifacts."""
        self._apply(self._trainer.train())

    def model(self) -> tuple[NearestNeighbors, NearestNeighbors]:
        """Returns the fitted (user_model, movie_model) pair."""
        return self._user_model, self._movie_model

    def recommend(self, *, user_id: int | None = None, movie_title: str | None = None) -> list[dict]:
        """Dispatches to the user-based or item-based path depending on which
        identifier is supplied."""
        if (user_id is None) == (movie_title is None):
            raise ValueError("provide exactly one of user_id or movie_title")
        if user_id is not None:
            return recommend_user_based(
                user_id,
                self.user_feature_df,
                self._user_model,
                self._movie_catalog,
                self._similar_users_count,
                self._top_n_recommendations,
            )
        return recommend_item_based(
            movie_title,
            self.movie_feature_df,
            self._movie_model,
            self._movie_catalog,
            self._top_n_recommendations,
        )

    def visualize(self) -> list[Path]:
        """Generates the report figures (spec 13.6) from the currently
        loaded dataset."""
        return generate_all_figures(self._merged_dataset, self._figures_dir, self._top_n_recommendations)

    def known_user_ids(self) -> list[int]:
        """User IDs present in the trained user-movie matrix, for input
        validation (spec 13.5)."""
        return list(self.user_feature_df.index)

    def recommendable_movie_titles(self) -> list[str]:
        """The filtered, recommendable movie set (spec 13.5), for input
        validation and the /api/movies autocomplete source."""
        return sorted(self._movie_catalog.index)

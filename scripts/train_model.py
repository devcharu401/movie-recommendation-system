import logging
from pathlib import Path

import joblib

from app import create_app
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

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"


def main() -> None:
    app = create_app()
    with app.app_context():
        merged = build_preprocessed_dataset(app.config["MIN_RATINGS_PER_MOVIE"])

        user_movie_matrix = build_user_movie_matrix(merged)
        user_feature_df = build_user_features(user_movie_matrix)
        movie_feature_df = build_movie_features(user_movie_matrix)
        movie_catalog = build_movie_catalog(merged)

        user_model = fit_nearest_neighbors(to_sparse_matrix(user_feature_df))
        movie_model = fit_nearest_neighbors(to_sparse_matrix(movie_feature_df))

        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        artifacts = {
            "user_feature_df.joblib": user_feature_df,
            "movie_feature_df.joblib": movie_feature_df,
            "movie_catalog.joblib": movie_catalog,
            "user_model.joblib": user_model,
            "movie_model.joblib": movie_model,
        }
        for filename, artifact in artifacts.items():
            path = MODELS_DIR / filename
            joblib.dump(artifact, path)
            logger.info("wrote %s", path)


if __name__ == "__main__":
    main()

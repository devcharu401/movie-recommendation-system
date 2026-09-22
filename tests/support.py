from __future__ import annotations

from app import create_app
from app.config import Config
from app.services.model_trainer import ModelArtifacts, ModelTrainer
from app.services.recommendation_service import Recommendation

PRECONDITIONS_MESSAGE = (
    "Model artifacts or the database are missing. Run scripts/load_database.py "
    "and scripts/train_model.py before running the test suite."
)


class MissingPreconditionsError(RuntimeError):
    """Raised when the database or the trained model artifacts a test
    class needs are not present. A real error, not unittest.SkipTest —
    a suite run without these preconditions must report FAILED, not OK
    with a skip count that reads as a pass."""


def load_model_artifacts() -> ModelArtifacts:
    """Loads the trained KNN artifacts directly through ModelTrainer, with
    no Flask app and no database involved — the ML layer is
    framework-agnostic (spec 14), so its tests should not need either."""
    trainer = ModelTrainer(Config.MODELS_DIR, Config.MIN_RATINGS_PER_MOVIE)
    try:
        return trainer.load()
    except FileNotFoundError as exc:
        raise MissingPreconditionsError(f"{PRECONDITIONS_MESSAGE} ({exc})") from exc


def build_offline_service() -> Recommendation:
    """Builds a Recommendation service straight from the trained model
    artifacts, without a Flask app or a database connection. Every method
    that does not read viewer facts from the database — known_user_ids,
    recommendable_movie_titles, browse_genres, browse_by_genre,
    performance_report — works from this alone."""
    trainer = ModelTrainer(Config.MODELS_DIR, Config.MIN_RATINGS_PER_MOVIE)
    try:
        return Recommendation(
            trainer,
            Config.SIMILAR_USERS_COUNT,
            Config.TOP_N_RECOMMENDATIONS,
            Config.FIGURES_DIR,
            Config.GENERATED_DIR,
        )
    except RuntimeError as exc:
        raise MissingPreconditionsError(f"{PRECONDITIONS_MESSAGE} ({exc})") from exc


def load_app_and_service():
    """Builds the Flask app and its recommendation service for test
    classes that also need the database — dataset statistics, a viewer's
    profile, the genre affinity chart, and every route. Raises
    MissingPreconditionsError with a clear instruction if the database
    has not been loaded or the model has not been trained."""
    try:
        app = create_app("development")
        with app.app_context():
            app.extensions["recommendation_service"].dataset_statistics()
    except Exception as exc:
        raise MissingPreconditionsError(f"{PRECONDITIONS_MESSAGE} ({exc})") from exc
    return app, app.extensions["recommendation_service"]

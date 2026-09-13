from flask import Flask

from app.api.errors import register_error_handlers
from app.api.routes import bp as api_bp
from app.config import config_by_name
from app.extensions import db
from app.services.model_trainer import ModelTrainer
from app.services.recommendation_service import Recommendation
from app.utils.logging import configure_logging


def create_app(config_name: str = "development", load_recommendation_service: bool = True) -> Flask:
    """load_recommendation_service is False for scripts that only need
    database access (load_database.py, train_model.py, evaluate.py) —
    those run before trained model artifacts necessarily exist, so eagerly
    building the service here would fail on a clean checkout."""
    app = Flask(__name__)
    app.config.from_object(config_by_name[config_name])

    if config_name == "production" and not app.config["SECRET_KEY"]:
        raise RuntimeError("SECRET_KEY must be set in the environment when running with the production config")

    configure_logging(app.config["LOG_LEVEL"])
    db.init_app(app)

    if load_recommendation_service:
        trainer = ModelTrainer(app.config["MODELS_DIR"], app.config["MIN_RATINGS_PER_MOVIE"])
        app.extensions["recommendation_service"] = Recommendation(
            trainer,
            app.config["SIMILAR_USERS_COUNT"],
            app.config["TOP_N_RECOMMENDATIONS"],
            app.config["FIGURES_DIR"],
        )

    app.register_blueprint(api_bp)
    register_error_handlers(app)

    return app

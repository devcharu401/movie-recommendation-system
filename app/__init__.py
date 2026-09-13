from flask import Flask

from app.config import config_by_name
from app.extensions import db
from app.utils.logging import configure_logging


def create_app(config_name: str = "development") -> Flask:
    app = Flask(__name__)
    app.config.from_object(config_by_name[config_name])

    configure_logging(app.config["LOG_LEVEL"])
    db.init_app(app)

    return app

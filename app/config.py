import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")


class Config:
    APP_ENV = os.environ.get("APP_ENV", "development")
    SECRET_KEY = os.environ.get("SECRET_KEY", "")
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "DATABASE_URL", f"sqlite:///{BASE_DIR / 'data' / 'movie_recommendation.db'}"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
    MIN_RATINGS_PER_MOVIE = int(os.environ.get("MIN_RATINGS_PER_MOVIE", 100))
    TOP_N_RECOMMENDATIONS = int(os.environ.get("TOP_N_RECOMMENDATIONS", 10))
    SIMILAR_USERS_COUNT = int(os.environ.get("SIMILAR_USERS_COUNT", 5))


class DevelopmentConfig(Config):
    DEBUG = True


class ProductionConfig(Config):
    DEBUG = False


config_by_name = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
}

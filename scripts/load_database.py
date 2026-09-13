import logging
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from app import create_app
from app.extensions import db
from app.ml.data_loader import load_movie_data, load_ratings_data, load_user_data, validate_data_format
from app.models import Movie, Rating, User

logger = logging.getLogger(__name__)

RAW_DIR = BASE_DIR / "data" / "raw" / "ml-100k"


def main() -> None:
    app = create_app()
    with app.app_context():
        users_df = load_user_data(RAW_DIR / "u.user")
        movies_df = load_movie_data(RAW_DIR / "u.item")
        ratings_df = load_ratings_data(RAW_DIR / "u.data")
        validate_data_format(users_df, movies_df, ratings_df)

        # drop_all/create_all rather than delete-and-reinsert: keeps re-runs
        # idempotent even after a schema change during development.
        db.drop_all()
        db.create_all()

        db.session.bulk_insert_mappings(User, users_df.to_dict("records"))
        db.session.bulk_insert_mappings(Movie, movies_df.to_dict("records"))
        db.session.bulk_insert_mappings(Rating, ratings_df.to_dict("records"))
        db.session.commit()

        logger.info("loaded %d users", len(users_df))
        logger.info("loaded %d movies", len(movies_df))
        logger.info("loaded %d ratings", len(ratings_df))


if __name__ == "__main__":
    main()

from app.extensions import db
from app.models import Rating


def get_all_ratings() -> list[dict]:
    """Read-only access to the ratings table (spec 15: no write path exposed)."""
    ratings = db.session.query(Rating).all()
    return [
        {
            "rating_id": rating.rating_id,
            "user_id": rating.user_id,
            "movie_id": rating.movie_id,
            "rating": rating.rating,
            "timestamp": rating.timestamp,
        }
        for rating in ratings
    ]

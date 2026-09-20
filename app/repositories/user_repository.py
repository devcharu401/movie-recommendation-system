from sqlalchemy import func

from app.extensions import db
from app.models import Rating, User


def get_all_users() -> list[dict]:
    """Read-only access to the users table (spec 15: no write path exposed)."""
    users = db.session.query(User).all()
    return [
        {
            "user_id": user.user_id,
            "age": user.age,
            "gender": user.gender,
            "occupation": user.occupation,
            "zip_code": user.zip_code,
        }
        for user in users
    ]


def get_user_profile(user_id: int) -> dict | None:
    """Profile facts for one viewer (spec 13.5): age and occupation come
    straight off the row; films rated and mean rating are an aggregate over
    their ratings rather than a Python count over loaded rows. Returns None
    if user_id doesn't exist."""
    user = db.session.query(User).filter(User.user_id == user_id).one_or_none()
    if user is None:
        return None

    films_rated, mean_rating = (
        db.session.query(func.count(Rating.rating_id), func.avg(Rating.rating))
        .filter(Rating.user_id == user_id)
        .one()
    )

    return {
        "age": user.age,
        "occupation": user.occupation,
        "films_rated": films_rated,
        "mean_rating": mean_rating,
    }

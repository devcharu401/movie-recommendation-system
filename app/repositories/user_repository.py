from app.extensions import db
from app.models import User


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

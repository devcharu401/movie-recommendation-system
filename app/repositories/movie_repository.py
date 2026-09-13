from app.extensions import db
from app.models import Movie


def get_all_movies() -> list[dict]:
    """Read-only access to the movies table (spec 15: no write path exposed)."""
    movies = db.session.query(Movie).all()
    return [
        {
            "movie_id": movie.movie_id,
            "movie_title": movie.movie_title,
            "release_date": movie.release_date,
            "genre": movie.genre,
        }
        for movie in movies
    ]

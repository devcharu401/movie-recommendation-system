from sqlalchemy import func

from app.extensions import db
from app.models import Movie, Rating, User


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


def get_dataset_statistics() -> dict:
    """Raw dataset counts for the landing page (spec 13.5), by aggregate SQL
    rather than loading full tables into Python. Spans all three tables, so
    it lives here rather than in user_repository or movie_repository —
    section 6 authorises no separate dataset repository.

    distinct_genres splits each movie's pipe-delimited genre string (spec
    3.4) and counts the distinct genre names that come out of it. Counting
    distinct values of the genre column directly would count distinct
    genre *combinations* (e.g. "Action|Comedy" as one value), which is a
    much larger and different figure than the number of genres."""
    total_viewers = db.session.query(func.count(User.user_id)).scalar()
    total_films = db.session.query(func.count(Movie.movie_id)).scalar()
    genre_combinations = [row[0] for row in db.session.query(Movie.genre).distinct()]
    distinct_genres = len(
        {genre for combination in genre_combinations if combination for genre in combination.split("|")}
    )
    total_ratings, earliest_rating, latest_rating = db.session.query(
        func.count(Rating.rating_id),
        func.min(Rating.timestamp),
        func.max(Rating.timestamp),
    ).one()

    return {
        "total_viewers": total_viewers,
        "total_films": total_films,
        "total_ratings": total_ratings,
        "distinct_genres": distinct_genres,
        "earliest_rating": earliest_rating,
        "latest_rating": latest_rating,
    }


def get_user_genre_distribution(user_id: int) -> list[dict]:
    """Per-genre share of one viewer's rated films (spec 13.5, viewer
    profile). Genres are stored pipe-delimited on the movie (spec 3.4), so
    the split happens here, against the small set of films this one user
    rated, rather than in the service layer or the template."""
    genre_combinations = [
        row[0]
        for row in db.session.query(Movie.genre).join(Rating, Rating.movie_id == Movie.movie_id).filter(
            Rating.user_id == user_id
        )
    ]
    total_films = len(genre_combinations)
    if total_films == 0:
        return []

    counts: dict[str, int] = {}
    for combination in genre_combinations:
        if not combination:
            continue
        for genre in combination.split("|"):
            counts[genre] = counts.get(genre, 0) + 1

    return [
        {"genre": genre, "share": count / total_films}
        for genre, count in sorted(counts.items(), key=lambda item: item[1], reverse=True)
    ]

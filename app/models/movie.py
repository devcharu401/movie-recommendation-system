from app.extensions import db


class Movie(db.Model):
    __tablename__ = "movies"

    movie_id = db.Column(db.Integer, primary_key=True)
    # Not unique: MovieLens lists some films, e.g. Chasing Amy (1997), under
    # two movie_ids.
    movie_title = db.Column(db.String(255), nullable=False)
    release_date = db.Column(db.Date)
    genre = db.Column(db.String(100), nullable=False)

    ratings = db.relationship("Rating", back_populates="movie")

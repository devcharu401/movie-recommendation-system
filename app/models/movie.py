from app.extensions import db


class Movie(db.Model):
    __tablename__ = "movies"

    movie_id = db.Column(db.Integer, primary_key=True)
    movie_title = db.Column(db.String(255))
    release_date = db.Column(db.Date)
    genre = db.Column(db.String(100))

    ratings = db.relationship("Rating", back_populates="movie")

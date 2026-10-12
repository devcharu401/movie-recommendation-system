from app.extensions import db


class Rating(db.Model):
    __tablename__ = "ratings"
    __table_args__ = (
        db.CheckConstraint("rating BETWEEN 1 AND 5", name="ck_ratings_rating_range"),
        db.UniqueConstraint("user_id", "movie_id", name="uq_ratings_user_movie"),
    )

    rating_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.user_id"), nullable=False, index=True)
    movie_id = db.Column(db.Integer, db.ForeignKey("movies.movie_id"), nullable=False, index=True)
    rating = db.Column(db.Integer, nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False)

    user = db.relationship("User", back_populates="ratings")
    movie = db.relationship("Movie", back_populates="ratings")

from app.extensions import db


class User(db.Model):
    __tablename__ = "users"

    user_id = db.Column(db.Integer, primary_key=True)
    age = db.Column(db.Integer)
    gender = db.Column(db.String(10))
    occupation = db.Column(db.String(100))
    zip_code = db.Column(db.String(10))

    ratings = db.relationship("Rating", back_populates="user")

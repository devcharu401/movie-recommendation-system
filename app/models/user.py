from app.extensions import db


class User(db.Model):
    __tablename__ = "users"
    __table_args__ = (
        db.CheckConstraint("age BETWEEN 1 AND 120", name="ck_users_age_range"),
        db.CheckConstraint("gender IN ('M', 'F')", name="ck_users_gender"),
    )

    user_id = db.Column(db.Integer, primary_key=True)
    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.String(10), nullable=False)
    occupation = db.Column(db.String(100), nullable=False)
    zip_code = db.Column(db.String(10), nullable=False)

    ratings = db.relationship("Rating", back_populates="user")

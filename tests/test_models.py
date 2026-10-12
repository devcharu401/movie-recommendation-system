from __future__ import annotations

import unittest
from datetime import datetime

from sqlalchemy.exc import IntegrityError

from app.extensions import db
from app.models import Movie, Rating, User
from tests import support

RATED_AT = datetime(1997, 12, 4, 15, 55, 49)


class SchemaConstraintTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = support.build_in_memory_database_app()

    def setUp(self):
        self.app_context = self.app.app_context()
        self.app_context.push()
        db.create_all()
        db.session.add(User(user_id=1, age=24, gender="M", occupation="technician", zip_code="85711"))
        db.session.add(Movie(movie_id=1, movie_title="Toy Story (1995)", genre="Animation|Children's|Comedy"))
        db.session.commit()

    def tearDown(self):
        db.session.rollback()
        db.drop_all()
        self.app_context.pop()

    def _assert_rejected(self, record):
        db.session.add(record)
        with self.assertRaises(IntegrityError):
            db.session.commit()

    def test_rating_zero_is_rejected(self):
        self._assert_rejected(Rating(user_id=1, movie_id=1, rating=0, timestamp=RATED_AT))

    def test_rating_six_is_rejected(self):
        self._assert_rejected(Rating(user_id=1, movie_id=1, rating=6, timestamp=RATED_AT))

    def test_duplicate_user_movie_pair_is_rejected(self):
        db.session.add(Rating(user_id=1, movie_id=1, rating=4, timestamp=RATED_AT))
        db.session.commit()
        self._assert_rejected(Rating(user_id=1, movie_id=1, rating=2, timestamp=RATED_AT))

    def test_rating_for_unknown_user_is_rejected(self):
        self._assert_rejected(Rating(user_id=999, movie_id=1, rating=4, timestamp=RATED_AT))

    def test_rating_for_unknown_movie_is_rejected(self):
        self._assert_rejected(Rating(user_id=1, movie_id=999, rating=4, timestamp=RATED_AT))

    def test_gender_outside_m_and_f_is_rejected(self):
        self._assert_rejected(User(user_id=2, age=30, gender="X", occupation="writer", zip_code="10003"))

    def test_age_zero_is_rejected(self):
        self._assert_rejected(User(user_id=2, age=0, gender="F", occupation="writer", zip_code="10003"))

    def test_null_movie_title_is_rejected(self):
        self._assert_rejected(Movie(movie_id=2, movie_title=None, genre="Drama"))

    def test_null_rating_timestamp_is_rejected(self):
        self._assert_rejected(Rating(user_id=1, movie_id=1, rating=4, timestamp=None))

    def test_duplicate_title_under_another_movie_id_is_accepted(self):
        db.session.add(Movie(movie_id=2, movie_title="Toy Story (1995)", genre="Animation|Children's|Comedy"))
        db.session.commit()
        self.assertEqual(Movie.query.filter_by(movie_title="Toy Story (1995)").count(), 2)


if __name__ == "__main__":
    unittest.main()

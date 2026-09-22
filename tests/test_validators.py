from __future__ import annotations

import unittest

from app.api.errors import ValidationError
from app.api.validators import validate_genre, validate_movie_title, validate_user_id
from tests import support


class ViewerIdValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = support.build_offline_service()
        cls.known_ids = cls.service.known_user_ids()
        cls.max_id = max(cls.known_ids)

    def test_valid_id_is_accepted(self):
        self.assertEqual(validate_user_id("10", self.known_ids), 10)

    def test_zero_is_rejected(self):
        with self.assertRaises(ValidationError):
            validate_user_id("0", self.known_ids)

    def test_highest_valid_id_is_accepted(self):
        self.assertEqual(validate_user_id(str(self.max_id), self.known_ids), self.max_id)

    def test_one_above_highest_is_rejected(self):
        with self.assertRaises(ValidationError):
            validate_user_id(str(self.max_id + 1), self.known_ids)

    def test_negative_id_is_rejected(self):
        with self.assertRaises(ValidationError):
            validate_user_id("-1", self.known_ids)

    def test_non_numeric_id_is_rejected(self):
        with self.assertRaises(ValidationError):
            validate_user_id("abc", self.known_ids)

    def test_empty_id_is_rejected(self):
        with self.assertRaises(ValidationError):
            validate_user_id("", self.known_ids)


class FilmTitleValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = support.build_offline_service()
        cls.known_titles = cls.service.recommendable_movie_titles()

    def test_valid_title_is_accepted(self):
        title = "Toy Story (1995)"
        self.assertEqual(validate_movie_title(title, self.known_titles), title)

    def test_unknown_title_is_rejected(self):
        with self.assertRaises(ValidationError):
            validate_movie_title("Not A Real Film (2099)", self.known_titles)

    def test_empty_title_is_rejected(self):
        with self.assertRaises(ValidationError):
            validate_movie_title("", self.known_titles)

    def test_title_with_apostrophe_is_accepted(self):
        title = "Schindler's List (1993)"
        self.assertEqual(validate_movie_title(title, self.known_titles), title)

    def test_title_with_comma_and_colon_is_accepted(self):
        title = "Godfather: Part II, The (1974)"
        self.assertEqual(validate_movie_title(title, self.known_titles), title)


class GenreValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = support.build_offline_service()
        cls.known_genres = [tile.name for tile in cls.service.browse_genres()]

    def test_valid_genre_is_accepted(self):
        self.assertEqual(validate_genre("Drama", self.known_genres), "Drama")

    def test_unknown_genre_is_rejected(self):
        with self.assertRaises(ValidationError):
            validate_genre("NotAGenre", self.known_genres)

    def test_childrens_genre_is_accepted(self):
        self.assertEqual(validate_genre("Children's", self.known_genres), "Children's")

    def test_film_noir_genre_is_accepted(self):
        self.assertEqual(validate_genre("Film-Noir", self.known_genres), "Film-Noir")


if __name__ == "__main__":
    unittest.main()

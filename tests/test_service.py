from __future__ import annotations

import unittest

from app.services.recommendation_service import _age_band
from tests import support


class DatasetStatisticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app, cls.service = support.load_app_and_service()
        cls.app_context = cls.app.app_context()
        cls.app_context.push()

    @classmethod
    def tearDownClass(cls):
        cls.app_context.pop()

    def test_raw_counts(self):
        stats = self.service.dataset_statistics()
        self.assertEqual(stats.raw_viewers, 943)
        self.assertEqual(stats.raw_films, 1682)
        self.assertEqual(stats.raw_ratings, 100000)

    def test_modelled_counts(self):
        self.assertEqual(len(self.service.recommendable_movie_titles()), 338)
        self.assertEqual(len(self.service._merged_dataset), 64819)

    def test_genre_count(self):
        stats = self.service.dataset_statistics()
        self.assertEqual(stats.raw_genres, 18)


class ModelledMatrixSparsityTests(unittest.TestCase):
    def test_sparsity_rounds_to_79_66_percent(self):
        report = support.load_evaluation_report()
        if report is None:
            self.skipTest("No persisted evaluation report; run scripts/train_model.py.")
        self.assertAlmostEqual(round(report.sparsity * 100, 2), 79.66)


class AgeBandBoundaryTests(unittest.TestCase):
    def test_under_18(self):
        self.assertEqual(_age_band(17), "Under 18")

    def test_eighteen_is_the_low_edge_of_18_24(self):
        self.assertEqual(_age_band(18), "18-24")

    def test_twenty_four_is_the_high_edge_of_18_24(self):
        self.assertEqual(_age_band(24), "18-24")

    def test_twenty_five_is_the_low_edge_of_25_34(self):
        self.assertEqual(_age_band(25), "25-34")

    def test_thirty_four_is_the_high_edge_of_25_34(self):
        self.assertEqual(_age_band(34), "25-34")

    def test_thirty_five_is_the_low_edge_of_35_44(self):
        self.assertEqual(_age_band(35), "35-44")

    def test_forty_four_is_the_high_edge_of_35_44(self):
        self.assertEqual(_age_band(44), "35-44")

    def test_forty_five_is_the_low_edge_of_45_54(self):
        self.assertEqual(_age_band(45), "45-54")

    def test_fifty_four_is_the_high_edge_of_45_54(self):
        self.assertEqual(_age_band(54), "45-54")

    def test_fifty_five_is_the_edge_of_55_and_over(self):
        self.assertEqual(_age_band(55), "55 and over")

    def test_well_above_fifty_five_is_still_55_and_over(self):
        self.assertEqual(_age_band(90), "55 and over")


class ViewerProfileWithoutOccupationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app, cls.service = support.load_app_and_service()
        cls.app_context = cls.app.app_context()
        cls.app_context.push()

    @classmethod
    def tearDownClass(cls):
        cls.app_context.pop()

    def test_viewer_57_has_no_recorded_occupation(self):
        profile = self.service.viewer_profile(57)
        self.assertIsNone(profile.occupation)


class BrowseGenreConsistencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = support.build_offline_service()

    def test_tile_counts_match_browse_by_genre_up_to_the_page_cap(self):
        page_size = self.service.browse_page_size()
        for tile in self.service.browse_genres():
            with self.subTest(genre=tile.name):
                films = self.service.browse_by_genre(tile.name)
                self.assertEqual(len(films), min(tile.film_count, page_size))

    def test_results_are_sorted_by_mean_rating_then_rating_count_then_title(self):
        # browse_by_genre() sorts on the unrounded mean rating; comparing
        # against BrowseFilm.mean_rating (rounded to 2dp for display) would
        # treat films as tied when they were not, and wrongly demand a
        # rating_count tiebreak between them.
        precise_mean_ratings = self.service._merged_dataset.groupby("movie_id")["rating"].mean()
        for tile in self.service.browse_genres():
            with self.subTest(genre=tile.name):
                films = self.service.browse_by_genre(tile.name)
                keys = [
                    (-precise_mean_ratings[film.movie_id], -film.rating_count, film.movie_title)
                    for film in films
                ]
                self.assertEqual(keys, sorted(keys))


class PersistedEvaluationTests(unittest.TestCase):
    def test_evaluation_report_exists(self):
        report = support.load_evaluation_report()
        self.assertIsNotNone(report)

    def test_every_metric_lies_between_zero_and_one(self):
        report = support.load_evaluation_report()
        if report is None:
            self.skipTest("No persisted evaluation report; run scripts/train_model.py.")
        self.assertGreaterEqual(report.sparsity, 0.0)
        self.assertLessEqual(report.sparsity, 1.0)
        for result in report.results:
            with self.subTest(neighbor_count=result.neighbor_count):
                self.assertGreaterEqual(result.precision_at_k, 0.0)
                self.assertLessEqual(result.precision_at_k, 1.0)
                self.assertGreaterEqual(result.recall_at_k, 0.0)
                self.assertLessEqual(result.recall_at_k, 1.0)
                self.assertGreaterEqual(result.coverage, 0.0)
                self.assertLessEqual(result.coverage, 1.0)


if __name__ == "__main__":
    unittest.main()

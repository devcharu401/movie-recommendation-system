from __future__ import annotations

import unittest

from app.config import Config
from app.ml.knn_engine import HIGH_RATING_THRESHOLD, recommend_item_based, recommend_user_based
from tests import support

SAMPLE_VIEWER_IDS = (1, 10, 57, 100, 405, 500, 750, 900)
SAMPLE_QUERY_TITLES = (
    "Toy Story (1995)",
    "Boot, Das (1981)",
    "Schindler's List (1993)",
)


def _recommend_user_based(artifacts, user_id):
    return recommend_user_based(
        user_id,
        artifacts.user_feature_df,
        artifacts.user_model,
        artifacts.movie_catalog,
        Config.SIMILAR_USERS_COUNT,
        Config.TOP_N_RECOMMENDATIONS,
    )


class NeighborSimilarityRangeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.artifacts = support.load_model_artifacts()

    def test_neighbor_similarity_is_between_zero_and_one(self):
        for user_id in SAMPLE_VIEWER_IDS:
            with self.subTest(user_id=user_id):
                _, neighbors, _, _ = _recommend_user_based(self.artifacts, user_id)
                for neighbor in neighbors:
                    self.assertGreaterEqual(neighbor.similarity, 0.0)
                    self.assertLessEqual(neighbor.similarity, 1.0)


class ContributingRatingThresholdTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.artifacts = support.load_model_artifacts()

    def test_every_contributing_rating_meets_the_high_rating_threshold(self):
        for user_id in SAMPLE_VIEWER_IDS:
            with self.subTest(user_id=user_id):
                _, _, _, contributing = _recommend_user_based(self.artifacts, user_id)
                for ratings in contributing.values():
                    for rating in ratings:
                        self.assertGreaterEqual(rating.rating, HIGH_RATING_THRESHOLD)


class RankScoreWorkedExampleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.artifacts = support.load_model_artifacts()

    def test_rank_score_equals_similarity_times_rating_summed_over_contributors(self):
        recommendations, neighbors, _, contributing = _recommend_user_based(self.artifacts, 10)
        target = next(r for r in recommendations if r["movie_title"] == "Nightmare Before Christmas, The (1993)")
        similarity_by_neighbor = {neighbor.user_id: neighbor.similarity for neighbor in neighbors}
        expected = sum(
            similarity_by_neighbor[rating.neighbor_id] * rating.rating
            for rating in contributing[target["movie_id"]]
        )
        self.assertAlmostEqual(target["rank_score"], expected, delta=1e-9)


class NeighborTopRatedOrderingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.artifacts = support.load_model_artifacts()

    def test_top_rated_titles_are_ordered_by_rating_then_rating_count(self):
        _, neighbors, _, _ = _recommend_user_based(self.artifacts, 10)
        catalog = self.artifacts.movie_catalog
        labels_by_title: dict[str, list[str]] = {}
        for label, row in catalog.iterrows():
            labels_by_title.setdefault(row.movie_title, []).append(label)

        for neighbor in neighbors:
            with self.subTest(neighbor_id=neighbor.user_id):
                ratings = self.artifacts.user_feature_df.loc[neighbor.user_id]
                keys = []
                for title in neighbor.top_rated_movies:
                    label = next(l for l in labels_by_title[title] if ratings.loc[l] > 0)
                    keys.append((-ratings.loc[label], -catalog.loc[label, "rating_count"]))
                self.assertEqual(keys, sorted(keys))

    def test_top_rated_titles_are_identical_across_two_calls(self):
        _, neighbors_first, _, _ = _recommend_user_based(self.artifacts, 10)
        _, neighbors_second, _, _ = _recommend_user_based(self.artifacts, 10)
        first = [neighbor.top_rated_movies for neighbor in neighbors_first]
        second = [neighbor.top_rated_movies for neighbor in neighbors_second]
        self.assertEqual(first, second)


class AlreadyRatedExclusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.artifacts = support.load_model_artifacts()

    def test_recommended_films_exclude_films_the_viewer_already_rated(self):
        for user_id in SAMPLE_VIEWER_IDS:
            with self.subTest(user_id=user_id):
                already_rated_labels = set(
                    self.artifacts.user_feature_df.columns[self.artifacts.user_feature_df.loc[user_id] > 0]
                )
                already_rated_ids = {
                    int(self.artifacts.movie_catalog.loc[label, "movie_id"]) for label in already_rated_labels
                }
                recommendations, _, _, _ = _recommend_user_based(self.artifacts, user_id)
                recommended_ids = {r["movie_id"] for r in recommendations}
                self.assertTrue(recommended_ids.isdisjoint(already_rated_ids))


class ItemBasedExcludesQueryFilmTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.artifacts = support.load_model_artifacts()

    def test_results_never_include_the_query_film(self):
        for title in SAMPLE_QUERY_TITLES:
            with self.subTest(title=title):
                results = recommend_item_based(
                    title,
                    self.artifacts.movie_feature_df,
                    self.artifacts.movie_model,
                    self.artifacts.movie_catalog,
                    Config.TOP_N_RECOMMENDATIONS,
                )
                titles = {r["movie_title"] for r in results}
                self.assertNotIn(title, titles)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.config import Config
from app.services.model_trainer import ModelTrainer
from app.services.recommendation_service import Recommendation
from tests import support


class GenreAffinityChartCachingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app, cls.reference_service = support.load_app_and_service()

    def test_chart_is_written_to_a_temporary_directory_and_reused_on_a_second_call(self):
        with tempfile.TemporaryDirectory() as tmp_dir, self.app.app_context():
            generated_dir = Path(tmp_dir)
            trainer = ModelTrainer(Config.MODELS_DIR, Config.MIN_RATINGS_PER_MOVIE)
            service = Recommendation(
                trainer,
                Config.SIMILAR_USERS_COUNT,
                Config.TOP_N_RECOMMENDATIONS,
                Config.FIGURES_DIR,
                generated_dir,
            )
            result = self.reference_service.recommend(user_id=10)

            first_path = service.genre_affinity_chart(10, result.recommendations)
            self.assertEqual(first_path.parent, generated_dir)
            self.assertTrue(first_path.exists())
            first_mtime = first_path.stat().st_mtime_ns

            second_path = service.genre_affinity_chart(10, result.recommendations)
            self.assertEqual(second_path, first_path)
            self.assertEqual(second_path.stat().st_mtime_ns, first_mtime)


if __name__ == "__main__":
    unittest.main()

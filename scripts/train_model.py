import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from app import create_app
from app.ml.evaluation import DEFAULT_NEIGHBOR_COUNTS, run_evaluation
from app.services.model_trainer import ModelTrainer

MODELS_DIR = BASE_DIR / "models"


def main() -> None:
    app = create_app(load_recommendation_service=False)
    with app.app_context():
        trainer = ModelTrainer(MODELS_DIR, app.config["MIN_RATINGS_PER_MOVIE"])
        artifacts = trainer.train()

        started = time.perf_counter()
        report = run_evaluation(
            artifacts.merged_dataset,
            neighbor_counts=DEFAULT_NEIGHBOR_COUNTS,
            top_n=app.config["TOP_N_RECOMMENDATIONS"],
        )
        elapsed = time.perf_counter() - started
        trainer.persist_evaluation(report)
        print(f"Evaluation added {elapsed:.1f}s to training.")


if __name__ == "__main__":
    main()

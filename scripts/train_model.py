import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from app import create_app
from app.services.model_trainer import ModelTrainer

MODELS_DIR = BASE_DIR / "models"


def main() -> None:
    app = create_app(load_recommendation_service=False)
    with app.app_context():
        trainer = ModelTrainer(MODELS_DIR, app.config["MIN_RATINGS_PER_MOVIE"])
        trainer.train()


if __name__ == "__main__":
    main()

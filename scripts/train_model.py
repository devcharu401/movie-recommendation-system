from pathlib import Path

from app import create_app
from app.services.model_trainer import ModelTrainer

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"


def main() -> None:
    app = create_app()
    with app.app_context():
        trainer = ModelTrainer(MODELS_DIR, app.config["MIN_RATINGS_PER_MOVIE"])
        trainer.train()


if __name__ == "__main__":
    main()

import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

SCRIPTS_DIR = Path(__file__).resolve().parent
STEPS = ["fetch_data.py", "load_database.py", "train_model.py"]


def _run_step(script_name: str) -> None:
    script_path = SCRIPTS_DIR / script_name
    logger.info("running %s", script_name)
    result = subprocess.run([sys.executable, str(script_path)])
    if result.returncode != 0:
        raise RuntimeError(f"{script_name} failed with exit code {result.returncode}; stopping")
    logger.info("%s completed", script_name)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for step in STEPS:
        _run_step(step)
    logger.info("initialisation complete")


if __name__ == "__main__":
    main()

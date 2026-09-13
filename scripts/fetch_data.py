import logging
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)

DATASET_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
DATASET_DIR = RAW_DIR / "ml-100k"
REQUIRED_FILES = ["u.user", "u.item", "u.data"]


def _already_fetched() -> bool:
    return all((DATASET_DIR / name).exists() for name in REQUIRED_FILES)


def _download(dest: Path) -> None:
    logger.info("downloading %s", DATASET_URL)
    try:
        urllib.request.urlretrieve(DATASET_URL, dest)
    except (urllib.error.URLError, OSError) as exc:
        raise RuntimeError(f"failed to download {DATASET_URL}: {exc}") from exc


def _extract(archive: Path, dest: Path) -> None:
    try:
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest)
    except zipfile.BadZipFile as exc:
        raise RuntimeError(f"{archive} is not a valid zip archive: {exc}") from exc


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if _already_fetched():
        logger.info("ml-100k already present at %s, skipping download", DATASET_DIR)
        return

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = RAW_DIR / "ml-100k.zip"

    if archive_path.exists():
        logger.info("archive already present at %s, skipping download", archive_path)
    else:
        _download(archive_path)

    _extract(archive_path, RAW_DIR)
    archive_path.unlink()

    if not _already_fetched():
        raise RuntimeError("extraction completed but expected dataset files are missing")

    logger.info("dataset ready at %s", DATASET_DIR)


if __name__ == "__main__":
    main()

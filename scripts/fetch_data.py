import hashlib
import logging
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)

DATASET_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
# Published at https://files.grouplens.org/datasets/movielens/ml-100k.zip.md5.
# ml-100k is a frozen historical snapshot (unlike ml-1m/ml-25m/ml-latest, it is
# never updated), so pinning the known-good checksum here is simpler and more
# reliable than fetching a second URL on every run to re-check it.
DATASET_MD5 = "0e33842e24a9c977be4e0107933c0723"
BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
DATASET_DIR = RAW_DIR / "ml-100k"
REQUIRED_FILES = ["u.user", "u.item", "u.data"]
ARCHIVE_MEMBERS = [f"ml-100k/{name}" for name in REQUIRED_FILES]


def _already_fetched() -> bool:
    return all((DATASET_DIR / name).exists() for name in REQUIRED_FILES)


def _download(dest: Path) -> None:
    logger.info("downloading %s", DATASET_URL)
    try:
        urllib.request.urlretrieve(DATASET_URL, dest)
    except (urllib.error.URLError, OSError) as exc:
        raise RuntimeError(f"failed to download {DATASET_URL}: {exc}") from exc


def _verify_checksum(archive: Path) -> None:
    digest = hashlib.md5(archive.read_bytes()).hexdigest()
    if digest != DATASET_MD5:
        raise RuntimeError(
            f"{archive} does not match the published checksum "
            f"(expected {DATASET_MD5}, got {digest}); the download may be corrupt"
        )


def _extract(archive: Path, dest: Path) -> None:
    # load_database.py only reads u.user, u.item and u.data; the archive also
    # carries the original MovieLens toolkit scripts and pre-split CV folds
    # this project doesn't use, so only the needed members are extracted.
    try:
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest, members=ARCHIVE_MEMBERS)
    except zipfile.BadZipFile as exc:
        raise RuntimeError(f"{archive} is not a valid zip archive: {exc}") from exc
    except KeyError as exc:
        raise RuntimeError(f"{archive} is missing an expected member: {exc}") from exc


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

    _verify_checksum(archive_path)
    _extract(archive_path, RAW_DIR)
    archive_path.unlink()

    if not _already_fetched():
        raise RuntimeError("extraction completed but expected dataset files are missing")

    logger.info("dataset ready at %s", DATASET_DIR)


if __name__ == "__main__":
    main()

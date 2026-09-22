from __future__ import annotations

import argparse
import logging
import os
import zipfile
from datetime import date
from pathlib import Path

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]
DIST_DIR = BASE_DIR / "dist"

EXCLUDED_DIR_NAMES = {".venv", "venv", ".git", "__pycache__", "dist", ".idea", ".vscode"}
EXCLUDED_RELATIVE_DIRS = {Path("app/static/generated")}
EXCLUDED_FILE_NAMES = {"CLAUDE.md", ".env"}
EXCLUDED_FILE_SUFFIXES = {".log"}

# Gated behind --include-data: the SQLite database, raw and processed
# MovieLens files (all under data/), and the trained model artifacts.
# Excluded by default because the dataset licence restricts redistribution.
DATA_RELATIVE_DIRS = {Path("data"), Path("models")}


def _is_excluded_dir(relative_dir: Path, include_data: bool) -> bool:
    if relative_dir.name in EXCLUDED_DIR_NAMES:
        return True
    if relative_dir in EXCLUDED_RELATIVE_DIRS:
        return True
    if not include_data and relative_dir in DATA_RELATIVE_DIRS:
        return True
    return False


def _is_excluded_file(relative_path: Path) -> bool:
    if relative_path.name in EXCLUDED_FILE_NAMES:
        return True
    if relative_path.suffix in EXCLUDED_FILE_SUFFIXES:
        return True
    return False


def collect_files(include_data: bool) -> list[Path]:
    collected: list[Path] = []
    for root, dirnames, filenames in os.walk(BASE_DIR):
        relative_root = Path(root).relative_to(BASE_DIR)
        dirnames[:] = [
            name for name in dirnames if not _is_excluded_dir(relative_root / name, include_data)
        ]

        for filename in filenames:
            relative_path = relative_root / filename
            if not _is_excluded_file(relative_path):
                collected.append(relative_path)

    return sorted(collected)


def build_zip(include_data: bool) -> Path:
    DIST_DIR.mkdir(parents=True, exist_ok=True)
    archive_name = f"reelmate-submission-{date.today():%Y%m%d}"
    if include_data:
        archive_name += "-with-data"
    zip_path = DIST_DIR / f"{archive_name}.zip"

    files = collect_files(include_data)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for relative_path in files:
            zf.write(BASE_DIR / relative_path, arcname=Path(archive_name) / relative_path)

    logger.info("wrote %s (%d files)", zip_path, len(files))
    return zip_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Builds a dated ZIP of the project for submission.")
    parser.add_argument(
        "--include-data",
        action="store_true",
        help=(
            "Include the SQLite database, trained model artifacts and raw MovieLens "
            "files. Excluded by default because the dataset licence restricts "
            "redistribution."
        ),
    )
    args = parser.parse_args()

    build_zip(args.include_data)


if __name__ == "__main__":
    main()

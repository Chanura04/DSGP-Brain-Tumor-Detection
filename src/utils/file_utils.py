"""
File path utilities.

Provides standard paths for organizing project data, including raw, interim,
and processed datasets.

Constants:
- PROJECT_ROOT (Path): The root directory of the project (two levels above this file).
- DATA_DIR (Path): Base data directory under the project root.
- RAW_DATA_DIR (Path): Directory for raw data files.
- INTERIM_DATA_DIR (Path): Directory for interim/temporary data files.
- PROCESSED_DATA_DIR (Path): Directory for processed outputs.
"""

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
INTERIM_DATA_DIR: Path = DATA_DIR / "interim"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
LABELED_IMAGES_DATA_DIR: Path = DATA_DIR / "labeled_images"

LOG_DIR: Path = Path("Logs")

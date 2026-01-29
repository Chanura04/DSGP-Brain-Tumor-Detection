"""
Logging utilities.

Provides constants and functions to configure logging consistently across the project.

Constants:
- LOG_FORMAT (str): Standard format string for log messages.

Functions:
- setup_logging(): Configures the root logger with INFO level, standard format,
  and logs to 'log.log' in append mode.
"""

import logging
from typing import Final
from src.utils.file_utils import LOG_DIR

LOG_FORMAT: Final[str] = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def setup_logging() -> None:
    """
    Set up the logging configuration for the project.

    Configures the root logger to:
    - Log at INFO level.
    - Use the standard log format.
    - Write logs to 'log.log' in append mode.
    :return:
    """
    logging.basicConfig(
        level=logging.INFO, format=LOG_FORMAT, filename=str(LOG_DIR / "log.log"), filemode="a"
    )

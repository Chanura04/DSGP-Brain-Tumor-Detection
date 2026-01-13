"""
General configuration constants.

This module defines general-purpose constants used throughout the project.

Constants:
- VALID_IMAGE_EXTENSIONS (set[str]): Set of allowed image file extensions.
  Example: {'.png', '.jpg', '.jpeg'}
"""

from typing import Set

VALID_IMAGE_EXTENSIONS: Set[str] = {".png", ".jpg", ".jpeg"}

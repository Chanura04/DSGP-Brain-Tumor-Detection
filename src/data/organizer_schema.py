from pathlib import Path
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
import os

from typing import List, Annotated

from src.data.config import DEFAULT_ORGANIZE_OUTPUT_DIR_NAME, DEFAULT_INCLUDE_MODE


class MoveImagesConfig(BaseModel):
    """
    Initialize the MoveImagesConfig instance.
    :param raw_dataset_path: Path to the raw dataset folder.
    :param interim_dataset_path: Path to the interim dataset folder.
    :param lookfor: List of folder names or classes to process.
    :param out: Name of output subdirectory. Defaults to DEFAULT_OUTPUT_DIR_NAME.
    :param include: If True, include subfolders matching folder names. Defaults to DEFAULT_INCLUDE_MODE.
    :param dry_run: If True, simulate copying without writing files. Defaults to False.
    :return: None
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    raw_dataset_path: Path
    interim_dataset_path: Path
    lookfor: Annotated[List[str], Field(min_length=1)]
    out: str = DEFAULT_ORGANIZE_OUTPUT_DIR_NAME
    include: bool = DEFAULT_INCLUDE_MODE
    dry_run: bool = False

    @field_validator("raw_dataset_path", mode="after")
    @classmethod
    def raw_path_must_exist(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"Raw dataset path does not exist: {v}")
        if not v.is_dir():
            raise ValueError(f"Raw dataset path must be a directory: {v}")
        return v.resolve()

    @field_validator("raw_dataset_path", mode="after")
    @classmethod
    def check_read_permission(cls, v: Path) -> Path:
        if not os.access(v, os.R_OK):
            raise ValueError(f"No read permission for {v}")
        return v.resolve()

    @field_validator("interim_dataset_path", mode="after")
    @classmethod
    def interim_path_valid(cls, v: Path) -> Path:
        # allow creation later, just ensure parent exists
        if not v.parent.exists():
            raise ValueError(
                f"Parent directory for interim dataset does not exist: {v.parent}"
            )
        return v.resolve()

    @field_validator("lookfor", mode="after")
    @classmethod
    def lookfor_not_empty_strings(cls, v: List[str]) -> List[str]:
        cleaned_words = [word.strip().lower() for word in v]
        if any(not word for word in cleaned_words):
            raise ValueError("lookfor cannot contain empty strings")
        return v

    @field_validator("out", mode="after")
    @classmethod
    def clean_out(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Output directory name cannot be empty")
        if "/" in v or "\\" in v:
            raise ValueError("Output directory name must not contain path separators")
        return v

    @model_validator(mode="after")
    def paths_match(self) -> "MoveImagesConfig":
        if self.raw_dataset_path.resolve() == self.interim_dataset_path.resolve():
            raise ValueError("DatasetIntegrityError: raw and interim paths must differ")
        return self

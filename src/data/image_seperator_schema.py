from pathlib import Path
import os
from pydantic import BaseModel, Field, field_validator, ConfigDict

from typing import Annotated

from src.data.config import DEFAULT_SEPARATOR_LOOKFOR_DIR_NAME, DEFAULT_SEPARATOR_OUTPUT_DIR_NAME, \
    DEFAULT_SEPARATOR_SOURCE_DIR_NAME, DEFAULT_SEPARATOR_APPLY_TO_DIR_NAME


class BaseImageSeparatorConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_path: Path
    lookfor: Annotated[str, Field(min_length=1)] = DEFAULT_SEPARATOR_LOOKFOR_DIR_NAME
    out: Annotated[str, Field(min_length=1)] = DEFAULT_SEPARATOR_OUTPUT_DIR_NAME
    dry_run: bool = False

    @field_validator("dataset_path", mode="after")
    @classmethod
    def dataset_path_must_exist(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"Dataset path does not exist: {v}")
        if not v.is_dir():
            raise ValueError(f"Dataset path must be a directory: {v}")
        return v.resolve()

    @field_validator("dataset_path", mode="after")
    @classmethod
    def check_read_permission(cls, v: Path) -> Path:
        if not os.access(v, os.R_OK):
            raise ValueError(f"No read permission for {v}")
        return v.resolve()

    @field_validator("lookfor", mode="after")
    @classmethod
    def clean_lookfor(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Look for directory name cannot be empty")
        if "/" in v or "\\" in v:
            raise ValueError("Look for directory name must not contain path separators")
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


class ClassificationImageSeparatorConfig(BaseImageSeparatorConfig):
    pass


class SegmentationImageSeparatorConfig(BaseImageSeparatorConfig):
    source: Annotated[Path, Field(min_length=1)] = DEFAULT_SEPARATOR_SOURCE_DIR_NAME
    apply_to: Annotated[Path, Field(min_length=1)] = DEFAULT_SEPARATOR_APPLY_TO_DIR_NAME

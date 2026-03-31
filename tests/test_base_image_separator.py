from pathlib import Path
import pytest
from PIL import Image

from src.data.base_image_separator import ImageSeparator


@pytest.fixture()
def image_separator():
    class DummyClass(ImageSeparator):
        def process_images(self) -> None:
            pass

        def filter_low_intensity_images(self) -> None:
            pass

    return DummyClass(dataset_path="data/interim/mri", lookfor="original", out="black", dry_run=True)


def test_str(image_separator):
    s = str(image_separator)
    assert "Separating Low Intensity Images " in s
    assert "(dry_run=True)" in s


def test_make_directory(tmp_path, image_separator):
    name = tmp_path / "glioma"

    expected = tmp_path / "glioma" / "black"
    result: Path = image_separator.make_directory(name)

    assert isinstance(result, Path)
    assert result == expected
    assert result.exists() is True
    assert result.is_dir() is True

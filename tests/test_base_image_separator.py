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


def test_is_mostly_black_false(tmp_path, image_separator):
    img_path = tmp_path / "glioma" / "original" / "img.jpg"

    img_path.parent.mkdir(parents=True)

    white_img = Image.new("RGB", (100, 100), color=(255, 255, 255))
    white_img.save(img_path)

    result = image_separator.is_mostly_black(img_path)

    assert result is False


def test_is_mostly_black_true(tmp_path, image_separator):
    img_path = tmp_path / "glioma" / "original" / "img.jpg"

    img_path.parent.mkdir(parents=True)

    white_img = Image.new("RGB", (100, 100), color=(0, 0, 0))
    white_img.save(img_path)

    result = image_separator.is_mostly_black(img_path)

    assert result is True


def test_is_mostly_black_when_image_missing(tmp_path, image_separator):
    img_path = tmp_path / "glioma" / "original" / "img.jpg"

    result = image_separator.is_mostly_black(img_path)

    assert result is True


def test_make_directory(tmp_path, image_separator):
    name = tmp_path / "glioma"

    expected = tmp_path / "glioma" / "black"
    result: Path = image_separator.make_directory(name)

    assert isinstance(result, Path)
    assert result == expected
    assert result.exists() is True
    assert result.is_dir() is True

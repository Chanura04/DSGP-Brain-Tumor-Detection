from pathlib import Path
from src.data.classification_image_separator import ClassificationImageSeparator
from src.data.image_seperator_schema import ClassificationImageSeparatorConfig

import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture()
def image_separator_config():
    return ClassificationImageSeparatorConfig(
        dataset_path="data/interim/mri", lookfor="original", out="black", dry_run=True)


@pytest.fixture()
def image_separator(image_separator_config):
    return ClassificationImageSeparator(image_separator_config)


def test_init(image_separator):
    assert image_separator.dataset_path == Path("data/interim/mri").resolve()
    assert image_separator.lookfor == "original"
    assert image_separator.out == "black"
    assert image_separator.dry_run is True


def test_repr(image_separator):
    r = repr(image_separator)
    assert "ClassificationImageSeparator(" in r
    assert "out=black)" in r


def test__process_single_image(tmp_path, image_separator, caplog):
    img_path = tmp_path / "glioma" / "original" / "img.jpg"
    dest_path = tmp_path / "glioma" / "no_black" / "img.jpg"

    with patch("src.data.base_image_separator.ImageSeparator.is_mostly_black", return_value=False):
        with caplog.at_level("INFO"):
            result = image_separator._process_single_image(img_path, dest_path)

    record = caplog.records[0]

    assert result is False
    assert record.levelname == "INFO"
    assert f"Copying {img_path} to {dest_path}" in caplog.text


def test__process_single_image_with_incorrect_path(tmp_path, image_separator, caplog):
    img_path = tmp_path / "glioma" / "original"
    dest_path = tmp_path / "glioma" / "no_black" / "img.jpg"

    image_separator.dry_run = False

    with patch("src.data.base_image_separator.ImageSeparator.is_mostly_black", return_value=False):
        with caplog.at_level("ERROR"):
            result = image_separator._process_single_image(img_path, dest_path)

    record = caplog.records[0]

    assert result is True
    assert record.levelname == "ERROR"
    assert "File processing failed" in record.message


def test_filter_low_intensity_images_with_non_existing_paths(tmp_path, image_separator, caplog):
    glioma = tmp_path / "glioma"
    pituitary = tmp_path / "pituitary"

    path_list = [glioma, pituitary]

    image_separator.source_folders = MagicMock(return_value=path_list)

    with caplog.at_level("DEBUG"):
        image_separator.filter_low_intensity_images()

    assert all(record.levelname == "DEBUG" for record in caplog.records)
    assert all(f"Skipping (no '{image_separator.lookfor}' folder): {path}" in record.message for record, path in
               zip(caplog.records, path_list))


def test_filter_low_intensity_images(tmp_path, image_separator, caplog):
    glioma = tmp_path / "glioma" / "original"
    pituitary = tmp_path / "pituitary" / "original"
    glioma.mkdir(parents=True)
    pituitary.mkdir(parents=True)

    img1 = glioma / "img1.jpg"
    img1.write_bytes(b"fake image data")
    img2 = pituitary / "img2.jpg"
    img2.write_bytes(b"fake image data")

    image_separator.source_folders = [tmp_path / "glioma", tmp_path / "pituitary"]

    with patch("src.data.base_image_separator.ImageSeparator.make_directory") as mock_make_dir:
        mock_make_dir.side_effect = lambda src: tmp_path / "output" / src.name

        with patch.object(image_separator, "_process_single_image", side_effect=[False, True]):
            with caplog.at_level("INFO"):
                image_separator.filter_low_intensity_images()

    logs = caplog.text
    assert "copied 1 images" in logs
    assert "removed 1 mostly black images" in logs

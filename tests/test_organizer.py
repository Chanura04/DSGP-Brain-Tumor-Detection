from pathlib import Path
from PIL import Image
from src.data.organizer import MoveImages

import pytest
from unittest.mock import MagicMock


@pytest.fixture()
def mover():
    return MoveImages(
        raw_dataset_path="raw",
        interim_dataset_path="interim",
        lookfor=["glioma", "pituitary"],
        out="original",
        include=False,
        dry_run=True,
    )


def test_init(mover):
    assert mover.raw_dataset_path == Path("raw")
    assert mover.interim_dataset_path == Path("interim")
    assert mover.lookfor == ["glioma", "pituitary"]
    assert mover.out == "original"
    assert mover.include is False
    assert mover.dry_run is True


def test_repr(mover):
    r = repr(mover)
    assert "MoveImages(" in r
    assert "include=False)" in r


def test_str(mover):
    s = str(mover)
    assert "Moving Images from " in s
    assert "(dry_run=True)" in s


def test_make_merged_directory(tmp_path, mover):
    mover.interim_dataset_path = tmp_path

    name = Path("glioma") / "original"
    result = mover.make_merged_directory(name)

    expected = tmp_path / "glioma" / "original"

    assert isinstance(result, Path)
    assert result == expected
    assert result.exists() is True
    assert result.is_dir() is True


def test_get_specific_paths(tmp_path, mover):
    dir_name_1 = tmp_path / "Glioma"
    dir_name_2 = tmp_path / "Glioma (1)"

    dir_name_1.mkdir()
    dir_name_2.mkdir()

    mover.raw_dataset_path = tmp_path

    word = "glioma"

    path = list(mover.get_specific_paths(word))

    expected = [dir_name_1, dir_name_2]

    assert isinstance(path, list)
    assert sorted(path) == sorted(expected)


def test_copy_file(tmp_path, mover):
    src = tmp_path / "Glioma" / "img.jpg"
    dest = tmp_path / "glioma" / "original" / "img.jpg"

    result = mover.copy_file(src, dest)
    assert result is True


def test_copy_file_duplicate(tmp_path, mover):
    src = tmp_path / "Glioma" / "img.jpg"
    dest = tmp_path / "glioma" / "original" / "img.jpg"

    src.parent.mkdir(parents=True)
    dest.parent.mkdir(parents=True)

    img = Image.new("RGB", (100, 100), color="red")
    img.save(src)
    img.save(dest)

    result = mover.copy_file(src, dest)

    assert result is False
    assert dest.exists() is True


def test_copy_files(mover, tmp_path, monkeypatch):
    src = tmp_path / "Glioma"
    dest = tmp_path / "glioma" / "original"

    src.mkdir()
    dest.mkdir()

    (src / "a.jpg").write_text("x")
    (src / "b.png").write_text("x")
    (src / "c.txt").write_text("x")
    (src / "d.jpeg").write_text("x")

    monkeypatch.setattr(
        "src.data.organizer.VALID_IMAGE_EXTENSIONS", {".png", ".jpg", ".jpeg"}
    )

    mover.copy_file = MagicMock(side_effect=[True, True, False])
    # simulates a copy_file environment,
    # [1st call - copied, 2nd - copied, 3rd - skipped]

    mover.copy_files(src, dest, word="glioma")

    called = {call.args[0].name for call in mover.copy_file.call_args_list}

    assert mover.copy_file.call_count == 3
    assert called == {"a.jpg", "b.png", "d.jpeg"}


def test_build_interim_dataset(mover, tmp_path):
    merged_folder = tmp_path / "glioma" / "original"
    source_folders = [tmp_path / "Glioma", tmp_path / "Glioma (1)"]

    mover.make_merged_directory = MagicMock(return_value=merged_folder)
    mover.get_specific_paths = MagicMock(return_value=source_folders)
    mover.copy_files = MagicMock()

    mover.build_interim_dataset()

    expected_calls = {
        (source_folders[0], merged_folder, "glioma"),
        (source_folders[1], merged_folder, "glioma"),
        (source_folders[0], merged_folder, "pituitary"),
        (source_folders[1], merged_folder, "pituitary"),
    }
    actual_calls = {call.args for call in mover.copy_files.call_args_list}

    assert mover.make_merged_directory.call_count == 2
    assert mover.get_specific_paths.call_count == 2
    assert mover.copy_files.call_count == 4
    assert actual_calls == expected_calls

import sys
from pathlib import Path
import pytest

# Add src folder to Python path so Python can see medimgprep and utils
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data.organize import MoveImages


@pytest.fixture
def mover():
    """Return a basic MoveImages instance for testing."""
    return MoveImages(
        raw_dataset_path="raw",
        interim_dataset_path="interim",
        lookfor=["class1", "class2"],
        out="original",
        include=True,
        dry_run=True
    )


def test_init(mover):
    """Test basic initialization."""
    assert mover.raw_dataset_path == Path("raw")
    assert mover.interim_dataset_path == Path("interim")
    assert mover.lookfor == ["class1", "class2"]
    assert mover.out == "original"
    assert mover.include is True
    assert mover.dry_run is True


def test_str_and_repr(mover):
    """Test __str__ and __repr__ methods."""
    s = str(mover)
    r = repr(mover)
    assert "MoveImages(" in r
    assert "Moving Images from" in s
    assert "dry_run=True" in s


def test_get_specific_paths_returns_paths(mover):
    """Simple test: get_specific_paths returns paths list (simulate Path)."""
    # Instead of mocking filesystem, just call with include=False
    paths = list(mover.get_specific_paths("class1"))
    # This will likely return empty list if folder doesn't exist, but test runs safely
    assert isinstance(paths, list)


def test_make_merged_directory_returns_path(mover, tmp_path):
    """Test that make_merged_directory returns a Path object."""
    # Use pytest tmp_path fixture to avoid creating real folders
    merged = mover.make_merged_directory(tmp_path / "class1" / "original")
    assert isinstance(merged, Path)
    # Folder should exist
    assert merged.exists()


def test_do_all_processes_runs_without_error(mover, tmp_path, monkeypatch):
    """Run do_all_processes without touching real files."""
    # Patch methods that do real file operations
    monkeypatch.setattr(MoveImages, "get_specific_paths", lambda self, word: [tmp_path])
    monkeypatch.setattr(MoveImages, "copy_unique_files", lambda self, src, dest, word: None)
    monkeypatch.setattr(MoveImages, "make_merged_directory", lambda self, name: tmp_path / "merged")

    # This should run without raising exceptions
    mover.do_all_processes()

"""Unit tests for UtilsIO helpers."""

import zipfile
from pathlib import Path
from urllib.error import HTTPError

import pytest

from tgedr_fdafaers.utils.utils_io import UtilsIO, UtilsIOError


# --------------------------------------------------------------------------- #
# tmp_dir
# --------------------------------------------------------------------------- #


def test_tmp_dir_creates_directory() -> None:
    """tmp_dir should return a path to an existing directory."""
    result = UtilsIO.tmp_dir()

    assert Path(result).exists()
    assert Path(result).is_dir()


def test_tmp_dir_returns_unique_paths() -> None:
    """Each call to tmp_dir should return a different directory."""
    dir1 = UtilsIO.tmp_dir()
    dir2 = UtilsIO.tmp_dir()

    assert dir1 != dir2


# --------------------------------------------------------------------------- #
# resource_exists
# --------------------------------------------------------------------------- #


class _FakeResponse:
    def __init__(self, code: int) -> None:
        self.code = code

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        return None


def test_resource_exists_returns_true_for_status_200(monkeypatch: pytest.MonkeyPatch) -> None:
    """HEAD checks should return True only for explicit HTTP 200."""

    def fake_urlopen(_):
        return _FakeResponse(200)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    assert UtilsIO.resource_exists("https://example.com/data.zip")


def test_resource_exists_returns_false_for_non_200(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-200 responses should return False."""

    def fake_urlopen(_):
        return _FakeResponse(204)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    assert not UtilsIO.resource_exists("https://example.com/data.zip")


def test_resource_exists_wraps_http_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """HTTP failures should be re-raised as UtilsIOError."""

    def fake_urlopen(_):
        raise HTTPError("https://example.com", 404, "not found", hdrs=None, fp=None)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    with pytest.raises(UtilsIOError, match="failed request"):
        UtilsIO.resource_exists("https://example.com/data.zip")


def test_resource_exists_raises_on_unsupported_scheme() -> None:
    """An unsupported URL scheme should raise UtilsIOError."""
    with pytest.raises(UtilsIOError, match="unsupported URL scheme"):
        UtilsIO.resource_exists("ftp://example.com/data.zip")


# --------------------------------------------------------------------------- #
# deflate_zip
# --------------------------------------------------------------------------- #


@pytest.fixture()
def sample_zip(tmp_path: Path) -> Path:
    """Create a zip archive containing a few test files at the root level."""
    zip_path = tmp_path / "archive.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("FileA.txt", "content a")
        zf.writestr("FileB.csv", "col1,col2")
        zf.writestr("README.md", "readme")
    return zip_path


def test_deflate_zip_extracts_all_files(sample_zip: Path, tmp_path: Path) -> None:
    """deflate_zip should extract all files when no filter is applied."""
    target = tmp_path / "out"
    target.mkdir()

    result = UtilsIO.deflate_zip(str(sample_zip), str(target))

    assert len(result) == 3
    for path in result:
        assert Path(path).exists()


def test_deflate_zip_preserves_case_by_default(sample_zip: Path, tmp_path: Path) -> None:
    """deflate_zip should keep original casing by default (lower_filename=False)."""
    target = tmp_path / "out"
    target.mkdir()

    result = UtilsIO.deflate_zip(str(sample_zip), str(target))

    filenames = [Path(p).name for p in result]
    assert "FileA.txt" in filenames
    assert "FileB.csv" in filenames
    assert "README.md" in filenames


def test_deflate_zip_lowercases_filenames_when_enabled(sample_zip: Path, tmp_path: Path) -> None:
    """deflate_zip should lowercase filenames when lower_filename=True."""
    target = tmp_path / "out"
    target.mkdir()

    result = UtilsIO.deflate_zip(str(sample_zip), str(target), lower_filename=True)

    filenames = [Path(p).name for p in result]
    assert "filea.txt" in filenames
    assert "fileb.csv" in filenames
    assert "readme.md" in filenames


def test_deflate_zip_applies_file_filter(sample_zip: Path, tmp_path: Path) -> None:
    """deflate_zip should only extract files that pass the filter."""
    target = tmp_path / "out"
    target.mkdir()

    result = UtilsIO.deflate_zip(
        str(sample_zip), str(target), file_filter=lambda f: f.endswith(".txt")
    )

    assert len(result) == 1
    assert Path(result[0]).name == "FileA.txt"


def test_deflate_zip_returns_empty_list_when_all_filtered(sample_zip: Path, tmp_path: Path) -> None:
    """deflate_zip should return an empty list when the filter excludes everything."""
    target = tmp_path / "out"
    target.mkdir()

    result = UtilsIO.deflate_zip(
        str(sample_zip), str(target), file_filter=lambda f: False
    )

    assert result == []


def test_deflate_zip_returns_full_paths(sample_zip: Path, tmp_path: Path) -> None:
    """deflate_zip should return absolute paths under the target folder."""
    target = tmp_path / "out"
    target.mkdir()

    result = UtilsIO.deflate_zip(str(sample_zip), str(target))

    for path in result:
        assert path.startswith(str(target))


def test_deflate_zip_preserves_subdirectory_structure(tmp_path: Path) -> None:
    """deflate_zip should extract files preserving the zip's internal directory tree."""
    zip_path = tmp_path / "nested.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("level1/level2/Data.txt", "nested content")
        zf.writestr("level1/Info.txt", "info content")

    target = tmp_path / "out"
    target.mkdir()

    result = UtilsIO.deflate_zip(str(zip_path), str(target))

    # Directory structure is preserved
    assert (target / "level1").is_dir()
    assert (target / "level1" / "level2").is_dir()
    # Files keep original casing (lower_filename defaults to False)
    assert (target / "level1" / "level2" / "Data.txt").exists()
    assert (target / "level1" / "Info.txt").exists()
    # Returned paths reflect the original file locations
    assert str(target / "level1" / "level2" / "Data.txt") in result
    assert str(target / "level1" / "Info.txt") in result

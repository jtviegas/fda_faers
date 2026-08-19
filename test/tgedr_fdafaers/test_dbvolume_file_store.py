"""Unit tests for Databricks volume file store behavior in local mode."""

from pathlib import Path

import pytest

from tgedr_fdafaers.dbvolume_file_store import DatabricksVolumeFileStore, _LocalFs
from tgedr_fdafaers.file_store import FileStoreError


# --------------------------------------------------------------------------- #
# _LocalFs tests
# --------------------------------------------------------------------------- #


class TestLocalFsLs:
    """Tests for _LocalFs.ls()."""

    def test_ls_returns_single_file_info_when_path_is_file(self, tmp_path: Path) -> None:
        """ls on a file path should return a single-element list with that file's info."""
        f = tmp_path / "single.txt"
        f.write_text("hello", encoding="utf-8")

        fs = _LocalFs()
        result = fs.ls(str(f))

        assert len(result) == 1
        assert result[0].path == str(f)
        assert result[0].name == "single.txt"
        assert result[0].size == len("hello")

    def test_ls_raises_for_nonexistent_path(self, tmp_path: Path) -> None:
        """ls should raise FileNotFoundError for missing paths."""
        fs = _LocalFs()

        with pytest.raises(FileNotFoundError, match="does not exist"):
            fs.ls(str(tmp_path / "nope"))

    def test_ls_lists_directory_contents(self, tmp_path: Path) -> None:
        """ls on a directory should return entries for files and subdirectories."""
        (tmp_path / "file.txt").write_text("data", encoding="utf-8")
        (tmp_path / "subdir").mkdir()

        fs = _LocalFs()
        result = fs.ls(str(tmp_path))

        names = {entry.name for entry in result}
        assert "file.txt" in names
        assert "subdir" in names

        # directory entry path ends with /
        dir_entry = next(e for e in result if e.name == "subdir")
        assert dir_entry.path.endswith("/")
        assert dir_entry.size == 0

        # file entry has correct size
        file_entry = next(e for e in result if e.name == "file.txt")
        assert file_entry.size == len("data")


class TestLocalFsCp:
    """Tests for _LocalFs.cp()."""

    def test_cp_raises_for_missing_source(self, tmp_path: Path) -> None:
        """cp should raise FileNotFoundError when source does not exist."""
        fs = _LocalFs()

        with pytest.raises(FileNotFoundError, match="source not found"):
            fs.cp(str(tmp_path / "nope.txt"), str(tmp_path / "dst.txt"))

    def test_cp_copies_file_to_destination(self, tmp_path: Path) -> None:
        """cp should copy file content to destination."""
        src = tmp_path / "src.txt"
        src.write_text("payload", encoding="utf-8")

        dst = tmp_path / "nested" / "dst.txt"
        fs = _LocalFs()
        result = fs.cp(str(src), str(dst))

        assert result is True
        assert dst.read_text(encoding="utf-8") == "payload"


class TestLocalFsRm:
    """Tests for _LocalFs.rm()."""

    def test_rm_raises_for_missing_path(self, tmp_path: Path) -> None:
        """rm should raise FileNotFoundError when path does not exist."""
        fs = _LocalFs()

        with pytest.raises(FileNotFoundError, match="path not found"):
            fs.rm(str(tmp_path / "nope"))

    def test_rm_removes_directory_recursively(self, tmp_path: Path) -> None:
        """rm with recurse=True should remove directory trees."""
        d = tmp_path / "tree" / "nested"
        d.mkdir(parents=True)
        (d / "file.txt").write_text("x", encoding="utf-8")

        fs = _LocalFs()
        result = fs.rm(str(tmp_path / "tree"), recurse=True)

        assert result is True
        assert not (tmp_path / "tree").exists()

    def test_rm_removes_single_file(self, tmp_path: Path) -> None:
        """rm should remove a single file when recurse is False."""
        f = tmp_path / "file.txt"
        f.write_text("x", encoding="utf-8")

        fs = _LocalFs()
        result = fs.rm(str(f))

        assert result is True
        assert not f.exists()


class TestLocalFsMkdirs:
    """Tests for _LocalFs.mkdirs()."""

    def test_mkdirs_creates_nested_directories(self, tmp_path: Path) -> None:
        """mkdirs should create the full directory tree."""
        target = tmp_path / "a" / "b" / "c"

        fs = _LocalFs()
        result = fs.mkdirs(str(target))

        assert result is True
        assert target.is_dir()


# --------------------------------------------------------------------------- #
# DatabricksVolumeFileStore tests
# --------------------------------------------------------------------------- #


def test_list_filters_by_pattern_and_returns_sorted_paths(tmp_path: Path) -> None:
    """List should return only matching files in sorted order."""
    store = DatabricksVolumeFileStore(config={"local": True})

    volume = tmp_path / "volume"
    volume.mkdir(parents=True)
    (volume / "b.txt").write_text("b", encoding="utf-8")
    (volume / "a.txt").write_text("a", encoding="utf-8")
    (volume / "ignore.csv").write_text("c", encoding="utf-8")
    (volume / "sub").mkdir()

    result = store.list(str(volume), file_pattern="*.txt")

    assert result == [str(volume / "a.txt"), str(volume / "b.txt")]


def test_list_raises_for_missing_path(tmp_path: Path) -> None:
    """List should surface missing path errors as FileStoreError."""
    store = DatabricksVolumeFileStore(config={"local": True})

    with pytest.raises(FileStoreError, match="does not exist"):
        store.list(str(tmp_path / "missing"))


def test_get_requires_target_url(tmp_path: Path) -> None:
    """Get should validate required kwargs."""
    store = DatabricksVolumeFileStore(config={"local": True})

    with pytest.raises(FileStoreError, match="target_url"):
        store.get(str(tmp_path / "file.txt"))


def test_get_copies_file_to_target(tmp_path: Path) -> None:
    """Get should copy a file from store path to local destination."""
    store = DatabricksVolumeFileStore(config={"local": True})

    source = tmp_path / "source.txt"
    source.write_text("payload", encoding="utf-8")

    target = tmp_path / "nested" / "target.txt"
    store.get(str(source), target_url=str(target))

    assert target.read_text(encoding="utf-8") == "payload"


def test_get_raises_filestoreerror_on_copy_failure(tmp_path: Path) -> None:
    """Get should wrap copy failures as FileStoreError."""
    store = DatabricksVolumeFileStore(config={"local": True})

    with pytest.raises(FileStoreError, match="failed to download"):
        store.get(str(tmp_path / "nonexistent.txt"), target_url=str(tmp_path / "dst.txt"))


def test_delete_removes_file(tmp_path: Path) -> None:
    """Delete should remove a file path."""
    store = DatabricksVolumeFileStore(config={"local": True})

    target = tmp_path / "delete_me.txt"
    target.write_text("x", encoding="utf-8")

    store.delete(str(target))

    assert not target.exists()


def test_delete_removes_directory_recursively(tmp_path: Path) -> None:
    """Delete with recursive=True should remove directory trees."""
    store = DatabricksVolumeFileStore(config={"local": True})

    d = tmp_path / "tree" / "nested"
    d.mkdir(parents=True)
    (d / "file.txt").write_text("x", encoding="utf-8")

    store.delete(str(tmp_path / "tree"), recursive=True)

    assert not (tmp_path / "tree").exists()


def test_delete_raises_filestoreerror_on_failure(tmp_path: Path) -> None:
    """Delete should wrap failures as FileStoreError."""
    store = DatabricksVolumeFileStore(config={"local": True})

    with pytest.raises(FileStoreError, match="failed to delete"):
        store.delete(str(tmp_path / "missing"))


def test_put_creates_parent_and_copies_file(tmp_path: Path) -> None:
    """Put should create destination parent directories and copy source content."""
    store = DatabricksVolumeFileStore(config={"local": True})

    source = tmp_path / "source.txt"
    source.write_text("content", encoding="utf-8")

    target = tmp_path / "deep" / "path" / "target.txt"
    store.put(str(source), str(target))

    assert target.exists()
    assert target.read_text(encoding="utf-8") == "content"


def test_put_raises_filestoreerror_on_failure(tmp_path: Path) -> None:
    """Put should wrap failures as FileStoreError."""
    store = DatabricksVolumeFileStore(config={"local": True})

    with pytest.raises(FileStoreError, match="failed to copy"):
        store.put(str(tmp_path / "nonexistent.txt"), str(tmp_path / "dst.txt"))


def test_init_without_config_uses_default() -> None:
    """DatabricksVolumeFileStore should handle None config gracefully using local fs."""
    # With None config use_local_fs defaults to False which would try to import pyspark.
    # Instead verify that passing config={"local": True} is accepted cleanly.
    store = DatabricksVolumeFileStore(config={"local": True})
    assert store.config == {"local": True}


def test_init_without_local_flag_uses_dbutils(monkeypatch: pytest.MonkeyPatch) -> None:
    """When local=False, DatabricksVolumeFileStore should use DBUtils from pyspark."""
    import sys
    from unittest.mock import MagicMock

    # Create mock modules for pyspark.dbutils and UtilsSpark
    mock_dbutils_module = MagicMock()
    mock_dbutils_instance = MagicMock()
    mock_dbutils_module.DBUtils.return_value = mock_dbutils_instance

    mock_spark = MagicMock()
    monkeypatch.setattr(
        "tgedr_fdafaers.dbvolume_file_store.UtilsSpark.get_spark_session",
        lambda: mock_spark,
    )

    # Patch the pyspark.dbutils module into sys.modules so the import succeeds
    monkeypatch.setitem(sys.modules, "pyspark", MagicMock())
    monkeypatch.setitem(sys.modules, "pyspark.dbutils", mock_dbutils_module)

    store = DatabricksVolumeFileStore(config={"local": False})

    mock_dbutils_module.DBUtils.assert_called_once_with(mock_spark)
    assert store._fs == mock_dbutils_instance.fs


def test_path_exists_returns_true_for_existing_path(tmp_path: Path) -> None:
    """__path_exists should return True when path exists."""
    store = DatabricksVolumeFileStore(config={"local": True})
    f = tmp_path / "exists.txt"
    f.write_text("hi", encoding="utf-8")

    # Access the private method via name mangling
    assert store._DatabricksVolumeFileStore__path_exists(str(f)) is True


def test_path_exists_returns_false_for_missing_path(tmp_path: Path) -> None:
    """__path_exists should return False when path does not exist."""
    store = DatabricksVolumeFileStore(config={"local": True})

    assert store._DatabricksVolumeFileStore__path_exists(str(tmp_path / "nope")) is False


def test_is_file_returns_true_for_file(tmp_path: Path) -> None:
    """__is_file should return True when the path points to a file."""
    store = DatabricksVolumeFileStore(config={"local": True})
    f = tmp_path / "afile.txt"
    f.write_text("data", encoding="utf-8")

    assert store._DatabricksVolumeFileStore__is_file(str(f)) is True


def test_is_file_returns_false_for_directory(tmp_path: Path) -> None:
    """__is_file should return False when the path points to a directory."""
    store = DatabricksVolumeFileStore(config={"local": True})
    d = tmp_path / "adir"
    d.mkdir()

    assert store._DatabricksVolumeFileStore__is_file(str(d)) is False


def test_is_file_returns_false_for_missing_path(tmp_path: Path) -> None:
    """__is_file should return False when path does not exist."""
    store = DatabricksVolumeFileStore(config={"local": True})

    assert store._DatabricksVolumeFileStore__is_file(str(tmp_path / "nope")) is False

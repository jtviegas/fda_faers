"""Unit tests for low-level I/O URL existence checks."""

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

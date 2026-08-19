"""Unit tests for low-level I/O URL existence checks."""

from urllib.error import HTTPError

import pytest

from tgedr_fdafaers.utils.utils_io import UtilsIO, UtilsIOError


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

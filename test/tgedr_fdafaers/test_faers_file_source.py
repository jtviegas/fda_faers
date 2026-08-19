"""Unit tests for the FAERS source adapter."""

from pathlib import Path
from urllib.error import HTTPError

import pytest
from tgedr_dataops_abs.source import SourceException

from tgedr_fdafaers.faers_file_source import FaersFileSource


def test_list_returns_url_when_remote_resource_exists(monkeypatch: pytest.MonkeyPatch) -> None:
    """List should resolve and return the URL for a valid available period."""
    source = FaersFileSource()

    monkeypatch.setattr("tgedr_fdafaers.faers_file_source.UtilsIO.resource_exists", lambda _: True)

    result = source.list({"period": "24q1"})

    assert result == "https://fis.fda.gov/content/Exports/faers_ascii_2024Q1.zip"


def test_list_returns_none_when_remote_resource_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """List should return None when the period URL does not exist."""
    source = FaersFileSource()

    monkeypatch.setattr("tgedr_fdafaers.faers_file_source.UtilsIO.resource_exists", lambda _: False)

    result = source.list({"period": "24q1"})

    assert result is None


def test_list_raises_when_period_is_missing() -> None:
    """List should fail fast when context does not provide a period."""
    source = FaersFileSource()

    with pytest.raises(SourceException, match="period"):
        source.list({})


def test_list_raises_when_context_is_none() -> None:
    """List should fail when context is None."""
    source = FaersFileSource()

    with pytest.raises(SourceException, match="period"):
        source.list(None)


def test_get_raises_when_output_url_is_missing() -> None:
    """Get should require an output directory in context."""
    source = FaersFileSource()

    with pytest.raises(SourceException, match="output_url"):
        source.get({"period": "24q1"})


def test_get_raises_when_context_is_none() -> None:
    """Get should fail when context is None."""
    source = FaersFileSource()

    with pytest.raises(SourceException, match="output_url"):
        source.get(None)


def test_get_downloads_file_and_returns_target_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Get should build the expected target and call urlretrieve once."""
    source = FaersFileSource()
    captured: dict[str, str] = {}

    def fake_urlretrieve(url: str, target: str) -> None:
        captured["url"] = url
        captured["target"] = target

    monkeypatch.setattr("urllib.request.urlretrieve", fake_urlretrieve)

    result = source.get({"period": "24q2", "output_url": str(tmp_path)})

    assert captured["url"] == "https://fis.fda.gov/content/Exports/faers_ascii_2024Q2.zip"
    assert captured["target"] == str(tmp_path / "2024q2.zip")
    assert result == str(tmp_path / "2024q2.zip")


def test_get_ignores_404_and_still_returns_target_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Get should treat 404 as non-fatal and still return the expected path."""
    source = FaersFileSource()

    def fake_urlretrieve(_: str, __: str) -> None:
        raise HTTPError("https://example.com", 404, "not found", hdrs=None, fp=None)

    monkeypatch.setattr("urllib.request.urlretrieve", fake_urlretrieve)

    result = source.get({"period": "24q3", "output_url": str(tmp_path)})

    assert result == str(tmp_path / "2024q3.zip")


def test_get_wraps_non_404_http_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Get should wrap non-404 HTTP failures in SourceException."""
    source = FaersFileSource()

    def fake_urlretrieve(_: str, __: str) -> None:
        raise HTTPError("https://example.com", 500, "error", hdrs=None, fp=None)

    monkeypatch.setattr("urllib.request.urlretrieve", fake_urlretrieve)

    with pytest.raises(SourceException, match="failed request"):
        source.get({"period": "24q3", "output_url": str(tmp_path)})


def test_get_wraps_generic_exceptions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Get should wrap generic download exceptions in SourceException."""
    source = FaersFileSource()

    def fake_urlretrieve(_: str, __: str) -> None:
        raise OSError("network failure")

    monkeypatch.setattr("urllib.request.urlretrieve", fake_urlretrieve)

    with pytest.raises(SourceException, match="failed request"):
        source.get({"period": "24q3", "output_url": str(tmp_path)})


def test_get_raises_on_unsupported_output_url_scheme(tmp_path: Path) -> None:
    """Get should reject output URLs with unsupported schemes."""
    source = FaersFileSource()

    with pytest.raises(SourceException, match="unsupported URL scheme"):
        source.get({"period": "24q1", "output_url": "ftp://remote/path"})

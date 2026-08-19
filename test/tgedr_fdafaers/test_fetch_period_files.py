"""Unit tests for the FetchPeriodFiles ETL."""

from unittest.mock import patch, MagicMock

from tgedr_fdafaers.etl.fetch_period_files import FetchPeriodFiles
from tgedr_fdafaers.faers_file_source import FaersFileSource


# --------------------------------------------------------------------------- #
# extract
# --------------------------------------------------------------------------- #


@patch("tgedr_fdafaers.etl.fetch_period_files.FaersFileSource")
@patch("tgedr_fdafaers.etl.fetch_period_files.UtilsIO.tmp_dir", return_value="/tmp/fake")
def test_extract_fetches_files_for_each_period(mock_tmp_dir, mock_source_cls) -> None:
    """extract should call source.get for each period in the comma-separated string."""
    mock_source = MagicMock()
    mock_source.get.side_effect = ["/tmp/fake/2024q1.zip", "/tmp/fake/2024q2.zip"]
    mock_source_cls.return_value = mock_source

    etl = FetchPeriodFiles(configuration={"periods": "24q1,24q2", "max_periods": 5})
    etl.extract()

    assert mock_source.get.call_count == 2
    assert "24q1" in etl._files_fetched
    assert "24q2" in etl._files_fetched


@patch("tgedr_fdafaers.etl.fetch_period_files.FaersFileSource")
@patch("tgedr_fdafaers.etl.fetch_period_files.UtilsIO.tmp_dir", return_value="/tmp/fake")
def test_extract_respects_max_periods_limit(mock_tmp_dir, mock_source_cls) -> None:
    """extract should only fetch up to max_periods periods."""
    mock_source = MagicMock()
    mock_source.get.return_value = "/tmp/fake/file.zip"
    mock_source_cls.return_value = mock_source

    etl = FetchPeriodFiles(configuration={"periods": "24q1,24q2,24q3", "max_periods": 2})
    etl.extract()

    assert mock_source.get.call_count == 2
    assert len(etl._files_fetched) == 2


@patch("tgedr_fdafaers.etl.fetch_period_files.FaersFileSource")
@patch("tgedr_fdafaers.etl.fetch_period_files.UtilsIO.tmp_dir", return_value="/tmp/fake")
def test_extract_strips_whitespace_from_periods(mock_tmp_dir, mock_source_cls) -> None:
    """extract should trim whitespace around period values."""
    mock_source = MagicMock()
    mock_source.get.return_value = "/tmp/fake/file.zip"
    mock_source_cls.return_value = mock_source

    etl = FetchPeriodFiles(configuration={"periods": " 24q1 , 24q2 ", "max_periods": 5})
    etl.extract()

    assert "24q1" in etl._files_fetched
    assert "24q2" in etl._files_fetched


@patch("tgedr_fdafaers.etl.fetch_period_files.FaersFileSource")
@patch("tgedr_fdafaers.etl.fetch_period_files.UtilsIO.tmp_dir", return_value="/tmp/fake")
def test_extract_single_period(mock_tmp_dir, mock_source_cls) -> None:
    """extract should handle a single period without commas."""
    mock_source = MagicMock()
    mock_source.get.return_value = "/tmp/fake/2024q1.zip"
    mock_source_cls.return_value = mock_source

    etl = FetchPeriodFiles(configuration={"periods": "24q1", "max_periods": 5})
    etl.extract()

    assert mock_source.get.call_count == 1
    assert "24q1" in etl._files_fetched


# --------------------------------------------------------------------------- #
# transform
# --------------------------------------------------------------------------- #


def test_transform_is_noop() -> None:
    """transform should do nothing (no state change)."""
    etl = FetchPeriodFiles()
    etl._files_fetched = {"24q1": "/tmp/fake/2024q1.zip"}

    etl.transform()

    # State should remain unchanged
    assert etl._files_fetched == {"24q1": "/tmp/fake/2024q1.zip"}


# --------------------------------------------------------------------------- #
# load
# --------------------------------------------------------------------------- #


def test_load_returns_sorted_comma_separated_periods() -> None:
    """load should return fetched period keys sorted and comma-separated."""
    etl = FetchPeriodFiles()
    etl._files_fetched = {"24q2": "/tmp/b.zip", "24q1": "/tmp/a.zip", "23q4": "/tmp/c.zip"}

    result = etl.load()

    assert result == "23q4,24q1,24q2"


def test_load_returns_empty_string_when_no_files_fetched() -> None:
    """load should return an empty string when no files were fetched."""
    etl = FetchPeriodFiles()
    etl._files_fetched = {}

    result = etl.load()

    assert result == ""


def test_load_single_period() -> None:
    """load should return a single period without commas."""
    etl = FetchPeriodFiles()
    etl._files_fetched = {"24q1": "/tmp/file.zip"}

    result = etl.load()

    assert result == "24q1"


# --------------------------------------------------------------------------- #
# end-to-end (extract + transform + load)
# --------------------------------------------------------------------------- #


@patch("tgedr_fdafaers.etl.fetch_period_files.FaersFileSource")
@patch("tgedr_fdafaers.etl.fetch_period_files.UtilsIO.tmp_dir", return_value="/tmp/fake")
def test_full_etl_pipeline(mock_tmp_dir, mock_source_cls) -> None:
    """Full pipeline should fetch files for periods and return them sorted."""
    mock_source = MagicMock()
    mock_source.get.side_effect = ["/tmp/fake/2024q2.zip", "/tmp/fake/2024q1.zip"]
    mock_source_cls.return_value = mock_source

    etl = FetchPeriodFiles(configuration={"periods": "24q2,24q1", "max_periods": 10})
    etl.extract()
    etl.transform()
    result = etl.load()

    assert result == "24q1,24q2"

"""Unit tests for the Periods2Process ETL."""

from unittest.mock import patch, MagicMock

import pandas as pd

from tgedr_fdafaers.etl.periods_to_process import Periods2Process
from tgedr_fdafaers.utils.faers_period import FaersPeriod, UtilsFaersPeriod


# --------------------------------------------------------------------------- #
# extract
# --------------------------------------------------------------------------- #


@patch("tgedr_fdafaers.etl.periods_to_process.HuggingFaceDatasetStore")
def test_extract_loads_existing_periods_from_bronze_dataset(mock_store_cls) -> None:
    """extract should populate _existing_periods from the bronze dataset's 'period' column."""
    mock_store = MagicMock()
    mock_store_cls.return_value = mock_store

    df = pd.DataFrame({"period": ["12q4", "13q1", "13q2"]})
    mock_result = MagicMock()
    mock_result.train = df
    mock_store.get.return_value = mock_result

    etl = Periods2Process(configuration={"bronze_dataset": "org/dataset"})
    etl.extract()

    mock_store.get.assert_called_once_with(key="org/dataset")
    assert etl._existing_periods == ["12q4", "13q1", "13q2"]


@patch("tgedr_fdafaers.etl.periods_to_process.HuggingFaceDatasetStore")
def test_extract_handles_missing_dataset_gracefully(mock_store_cls) -> None:
    """extract should leave _existing_periods empty when the dataset does not exist."""
    from tgedr_dataops.store.hf_dataset import NoStoreException

    mock_store = MagicMock()
    mock_store_cls.return_value = mock_store
    mock_store.get.side_effect = NoStoreException("not found")

    etl = Periods2Process(configuration={"bronze_dataset": "org/missing"})
    etl.extract()

    assert etl._existing_periods == []


@patch("tgedr_fdafaers.etl.periods_to_process.HuggingFaceDatasetStore")
def test_extract_handles_empty_dataframe(mock_store_cls) -> None:
    """extract should leave _existing_periods empty when the dataset is empty."""
    mock_store = MagicMock()
    mock_store_cls.return_value = mock_store

    mock_result = MagicMock()
    mock_result.train = pd.DataFrame()
    mock_store.get.return_value = mock_result

    etl = Periods2Process(configuration={"bronze_dataset": "org/empty"})
    etl.extract()

    assert etl._existing_periods == []


@patch("tgedr_fdafaers.etl.periods_to_process.HuggingFaceDatasetStore")
def test_extract_deduplicates_periods(mock_store_cls) -> None:
    """extract should return unique periods even when the dataset has duplicates."""
    mock_store = MagicMock()
    mock_store_cls.return_value = mock_store

    df = pd.DataFrame({"period": ["12q4", "12q4", "13q1"]})
    mock_result = MagicMock()
    mock_result.train = df
    mock_store.get.return_value = mock_result

    etl = Periods2Process(configuration={"bronze_dataset": "org/dataset"})
    etl.extract()

    assert sorted(etl._existing_periods) == ["12q4", "13q1"]


# --------------------------------------------------------------------------- #
# transform
# --------------------------------------------------------------------------- #


@patch.object(UtilsFaersPeriod, "get_all_faers_periods")
def test_transform_finds_missing_periods(mock_get_all) -> None:
    """transform should identify periods present in get_all_faers_periods but missing from extracted data."""
    mock_get_all.return_value = [
        FaersPeriod(2012, 4),
        FaersPeriod(2013, 1),
        FaersPeriod(2013, 2),
        FaersPeriod(2013, 3),
    ]

    etl = Periods2Process()
    etl._existing_periods = ["12q4", "13q1"]
    etl.transform()

    assert etl._periods_missing == ["13q2", "13q3"]


@patch.object(UtilsFaersPeriod, "get_all_faers_periods")
def test_transform_returns_empty_when_all_periods_exist(mock_get_all) -> None:
    """transform should produce an empty list when no periods are missing."""
    mock_get_all.return_value = [
        FaersPeriod(2012, 4),
        FaersPeriod(2013, 1),
    ]

    etl = Periods2Process()
    etl._existing_periods = ["12q4", "13q1"]
    etl.transform()

    assert etl._periods_missing == []


@patch.object(UtilsFaersPeriod, "get_all_faers_periods")
def test_transform_all_missing_when_no_existing_periods(mock_get_all) -> None:
    """transform should report all periods as missing when _existing_periods is empty."""
    mock_get_all.return_value = [
        FaersPeriod(2012, 4),
        FaersPeriod(2013, 1),
    ]

    etl = Periods2Process()
    etl._existing_periods = []
    etl.transform()

    assert etl._periods_missing == ["12q4", "13q1"]


# --------------------------------------------------------------------------- #
# load
# --------------------------------------------------------------------------- #


def test_load_returns_sorted_comma_separated_periods() -> None:
    """load should return a sorted, comma-separated string of missing periods."""
    etl = Periods2Process()
    etl._periods_missing = ["13q2", "12q4", "13q1"]

    result = etl.load()

    assert result == "12q4,13q1,13q2"


def test_load_returns_empty_string_when_no_missing_periods() -> None:
    """load should return an empty string when there are no missing periods."""
    etl = Periods2Process()
    etl._periods_missing = []

    result = etl.load()

    assert result == ""


def test_load_returns_single_period_without_comma() -> None:
    """load should handle a single missing period without a trailing comma."""
    etl = Periods2Process()
    etl._periods_missing = ["24q1"]

    result = etl.load()

    assert result == "24q1"


# --------------------------------------------------------------------------- #
# end-to-end (extract + transform + load)
# --------------------------------------------------------------------------- #


@patch.object(UtilsFaersPeriod, "get_all_faers_periods")
@patch("tgedr_fdafaers.etl.periods_to_process.HuggingFaceDatasetStore")
def test_full_etl_pipeline(mock_store_cls, mock_get_all) -> None:
    """Full pipeline should extract existing, compute missing, and return sorted string."""
    mock_store = MagicMock()
    mock_store_cls.return_value = mock_store

    df = pd.DataFrame({"period": ["12q4", "13q2"]})
    mock_result = MagicMock()
    mock_result.train = df
    mock_store.get.return_value = mock_result

    mock_get_all.return_value = [
        FaersPeriod(2012, 4),
        FaersPeriod(2013, 1),
        FaersPeriod(2013, 2),
        FaersPeriod(2013, 3),
    ]

    etl = Periods2Process(configuration={"bronze_dataset": "org/dataset"})
    etl.extract()
    etl.transform()
    result = etl.load()

    assert result == "13q1,13q3"

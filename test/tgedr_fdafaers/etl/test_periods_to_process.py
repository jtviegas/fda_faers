"""Unit tests for the Periods2Process ETL."""

from unittest.mock import patch, MagicMock

from datasets.exceptions import DatasetNotFoundError
from tgedr_dataops.store.hf_dataset import NoStoreException

from tgedr_fdafaers.etl.periods_to_process import Periods2Process
from tgedr_fdafaers.utils.faers_period import FaersPeriod, UtilsFaersPeriod


# --------------------------------------------------------------------------- #
# extract
# --------------------------------------------------------------------------- #


def _dataset(periods: list[str | None]) -> MagicMock:
    dataset = MagicMock()
    dataset.unique.return_value = periods
    return dataset


@patch("tgedr_fdafaers.etl.periods_to_process.load_dataset")
def test_extract_loads_existing_periods_from_bronze_dataset(mock_load_dataset) -> None:
    """extract should load every table and collect its available periods."""
    mock_load_dataset.return_value = _dataset(["12q4", "13q1", "13q2"])

    etl = Periods2Process(configuration={"dataset_prefix": "org/bronze_"})
    etl.extract()

    assert mock_load_dataset.call_count == 7
    assert sorted(etl._existing_periods) == ["12q4", "13q1", "13q2"]
    assert mock_load_dataset.call_args.kwargs == {"split": "train"}
    assert mock_load_dataset.return_value.unique.call_args.args == ("period",)


@patch("tgedr_fdafaers.etl.periods_to_process.load_dataset")
def test_extract_intersects_periods_across_tables(mock_load_dataset) -> None:
    """extract should return only periods present in every table."""
    rows_all = ["12q4", "13q1", "13q2"]
    rows_subset = ["12q4", "13q1"]
    mock_load_dataset.side_effect = [
        _dataset(rows_all),
        _dataset(rows_subset),
        *[_dataset(rows_all) for _ in range(5)],
    ]

    etl = Periods2Process(configuration={"dataset_prefix": "org/bronze_"})
    etl.extract()

    assert sorted(etl._existing_periods) == ["12q4", "13q1"]


@patch("tgedr_fdafaers.etl.periods_to_process.load_dataset")
def test_extract_skips_missing_datasets(mock_load_dataset) -> None:
    """extract should skip datasets that raise known not-found exceptions."""
    rows = ["12q4", "13q1"]
    mock_load_dataset.side_effect = [
        DatasetNotFoundError("not found"),
        _dataset(rows),
        _dataset(rows),
        _dataset(rows),
        _dataset(rows),
        _dataset(rows),
        _dataset(rows),
    ]

    etl = Periods2Process(configuration={"dataset_prefix": "org/bronze_"})
    etl.extract()

    assert sorted(etl._existing_periods) == ["12q4", "13q1"]


@patch("tgedr_fdafaers.etl.periods_to_process.load_dataset")
def test_extract_skips_legacy_store_not_found_exception(mock_load_dataset) -> None:
    """extract should continue to tolerate the store's not-found exception."""
    rows = ["12q4"]
    mock_load_dataset.side_effect = [
        NoStoreException("not found"),
        *[_dataset(rows) for _ in range(6)],
    ]

    etl = Periods2Process(configuration={"dataset_prefix": "org/bronze_"})
    etl.extract()

    assert etl._existing_periods == ["12q4"]


@patch("tgedr_fdafaers.etl.periods_to_process.load_dataset")
def test_extract_handles_empty_datasets(mock_load_dataset) -> None:
    """extract should leave _existing_periods empty when every dataset is empty."""
    mock_load_dataset.return_value = _dataset([])

    etl = Periods2Process(configuration={"dataset_prefix": "org/bronze_"})
    etl.extract()

    assert etl._existing_periods == []


@patch("tgedr_fdafaers.etl.periods_to_process.load_dataset")
def test_extract_ignores_null_and_deduplicates_periods(mock_load_dataset) -> None:
    """extract should ignore null period values and deduplicate the rest."""
    mock_load_dataset.return_value = _dataset(["12q4", "12q4", None, "13q1"])

    etl = Periods2Process(configuration={"dataset_prefix": "org/bronze_"})
    etl.extract()

    assert sorted(etl._existing_periods) == ["12q4", "13q1"]


@patch("tgedr_fdafaers.etl.periods_to_process.load_dataset")
def test_extract_uses_dataset_prefix_per_table(mock_load_dataset) -> None:
    """extract should load each table using the configured prefix and train split."""
    mock_load_dataset.return_value = _dataset(["12q4"])

    etl = Periods2Process(configuration={"dataset_prefix": "org/bronze_"})
    etl.extract()

    dataset_ids = [call.args[0] for call in mock_load_dataset.call_args_list]
    assert dataset_ids == [
        "org/bronze_demo",
        "org/bronze_drug",
        "org/bronze_indi",
        "org/bronze_outc",
        "org/bronze_reac",
        "org/bronze_rpsr",
        "org/bronze_ther",
    ]
    assert all(call.kwargs == {"split": "train"} for call in mock_load_dataset.call_args_list)


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
@patch("tgedr_fdafaers.etl.periods_to_process.load_dataset")
def test_full_etl_pipeline(mock_load_dataset, mock_get_all) -> None:
    """Full pipeline should extract existing, compute missing, and return sorted string."""
    mock_load_dataset.return_value = _dataset(["12q4", "13q2"])

    mock_get_all.return_value = [
        FaersPeriod(2012, 4),
        FaersPeriod(2013, 1),
        FaersPeriod(2013, 2),
        FaersPeriod(2013, 3),
    ]

    etl = Periods2Process(configuration={"dataset_prefix": "org/bronze_"})
    etl.extract()
    etl.transform()
    result = etl.load()

    assert result == "13q1,13q3"

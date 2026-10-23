"""Unit tests for the ComputeRowsMetrics ETL."""

from unittest.mock import MagicMock, patch

import pytest
from datasets.exceptions import DatasetNotFoundError
from tgedr_dataops_abs.etl import EtlException

from tgedr_fdafaers.constants import Constants
from tgedr_fdafaers.etl.compute_rows_metrics import ComputeRowsMetrics


# --------------------------------------------------------------------------- #
# extract
# --------------------------------------------------------------------------- #


def _streaming_dataset(rows: list[dict[str, str | None]]) -> MagicMock:
    dataset = MagicMock()
    dataset.select_columns.return_value = dataset
    dataset.__iter__.return_value = iter(rows)
    return dataset


@patch("tgedr_fdafaers.etl.compute_rows_metrics.load_dataset")
def test_extract_loads_each_table_dataset(mock_load_dataset) -> None:
    """extract should stream each train split and store its period values."""
    rows = [{"period": "24q1"}, {"period": "24q1"}, {"period": "24q2"}]
    datasets = [_streaming_dataset(rows) for _ in Constants.TABLES]
    mock_load_dataset.side_effect = datasets

    etl = ComputeRowsMetrics(configuration={"dataset_prefix": "org/faers_bronze_"})
    etl.extract()

    assert mock_load_dataset.call_count == len(Constants.TABLES)
    assert etl._data == {table: rows and [row["period"] for row in rows] for table in Constants.TABLES}
    assert all(call.kwargs == {"split": "train", "streaming": True} for call in mock_load_dataset.call_args_list)
    assert all(dataset.select_columns.call_args.args == (["period"],) for dataset in datasets)


@patch("tgedr_fdafaers.etl.compute_rows_metrics.load_dataset")
def test_extract_raises_on_missing_dataset(mock_load_dataset) -> None:
    """extract should translate Hugging Face not-found errors into EtlException."""
    mock_load_dataset.side_effect = DatasetNotFoundError("not found")

    etl = ComputeRowsMetrics(configuration={"dataset_prefix": "org/faers_bronze_"})
    with pytest.raises(EtlException, match="required dataset not found"):
        etl.extract()

    assert etl._data == {}


@patch("tgedr_fdafaers.etl.compute_rows_metrics.load_dataset")
def test_extract_raises_on_empty_dataset(mock_load_dataset) -> None:
    """extract should raise EtlException when a required dataset has no non-null periods."""
    mock_load_dataset.return_value = _streaming_dataset([])

    etl = ComputeRowsMetrics(configuration={"dataset_prefix": "org/faers_bronze_"})
    with pytest.raises(EtlException, match="no periods for dataset"):
        etl.extract()

    assert etl._data == {}


@patch("tgedr_fdafaers.etl.compute_rows_metrics.load_dataset")
def test_extract_ignores_null_periods_and_raises_if_none_remain(mock_load_dataset) -> None:
    """extract should omit null period values and fail if no usable period remains."""
    mock_load_dataset.return_value = _streaming_dataset([{"period": None}, {"period": None}])

    etl = ComputeRowsMetrics(configuration={"dataset_prefix": "org/faers_bronze_"})
    with pytest.raises(EtlException, match="no periods for dataset"):
        etl.extract()

    assert etl._data == {}


@patch("tgedr_fdafaers.etl.compute_rows_metrics.load_dataset")
def test_extract_uses_prefix_and_table_for_key(mock_load_dataset) -> None:
    """extract should build dataset IDs as '{prefix}{table}' and select the train split."""
    mock_load_dataset.side_effect = [_streaming_dataset([{"period": "24q1"}]) for _ in Constants.TABLES]

    etl = ComputeRowsMetrics(configuration={"dataset_prefix": "org/faers_"})
    etl.extract()

    expected_ids = [f"org/faers_{table}" for table in Constants.TABLES]
    actual_ids = [call.args[0] for call in mock_load_dataset.call_args_list]
    assert actual_ids == expected_ids
    assert all(call.kwargs == {"split": "train", "streaming": True} for call in mock_load_dataset.call_args_list)


# --------------------------------------------------------------------------- #
# transform
# --------------------------------------------------------------------------- #


def test_transform_computes_rows_per_period() -> None:
    """transform should count period values for each table."""
    etl = ComputeRowsMetrics()
    etl._data = {
        "reac": ["24q1", "24q1", "24q2"],
        "drug": ["24q1", "24q2", "24q2", "24q2"],
    }

    etl.transform()

    assert etl._rows == {
        "reac": {"24q1": 2, "24q2": 1},
        "drug": {"24q1": 1, "24q2": 3},
    }


def test_transform_with_no_data_yields_empty_rows() -> None:
    """transform should leave _rows empty when no data was extracted."""
    etl = ComputeRowsMetrics()
    etl._data = {}

    etl.transform()

    assert etl._rows == {}


def test_transform_coerces_period_keys_to_str() -> None:
    """transform should store period keys as strings for non-string values."""
    etl = ComputeRowsMetrics()
    etl._data = {"reac": [241, 241, 242]}

    etl.transform()

    assert etl._rows == {"reac": {"241": 2, "242": 1}}


# --------------------------------------------------------------------------- #
# load
# --------------------------------------------------------------------------- #


@patch("tgedr_fdafaers.etl.compute_rows_metrics.Metrics")
def test_load_records_gauge_metric_for_each_table_period(mock_metrics_cls) -> None:
    """load should call add_to_gauge once per (table, period) with the right name and attributes."""
    mock_metrics = MagicMock()
    mock_metrics_cls.instance.return_value = mock_metrics

    etl = ComputeRowsMetrics()
    etl._rows = {
        "reac": {"24q1": 2, "24q2": 1},
        "drug": {"24q1": 5},
    }

    result = etl.load()

    assert mock_metrics.add_to_gauge.call_count == 3
    mock_metrics.add_to_gauge.assert_any_call(
        ComputeRowsMetrics.METRIC_NAME, 2, {"table": "reac", "period": "24q1"}
    )
    mock_metrics.add_to_gauge.assert_any_call(
        ComputeRowsMetrics.METRIC_NAME, 1, {"table": "reac", "period": "24q2"}
    )
    mock_metrics.add_to_gauge.assert_any_call(
        ComputeRowsMetrics.METRIC_NAME, 5, {"table": "drug", "period": "24q1"}
    )
    assert result == "3 metric points recorded"


@patch("tgedr_fdafaers.etl.compute_rows_metrics.Metrics")
def test_load_returns_zero_points_when_no_rows(mock_metrics_cls) -> None:
    """load should record nothing and return '0 metric points recorded' when _rows is empty."""
    mock_metrics = MagicMock()
    mock_metrics_cls.instance.return_value = mock_metrics

    etl = ComputeRowsMetrics()
    etl._rows = {}

    result = etl.load()

    mock_metrics.add_to_gauge.assert_not_called()
    assert result == "0 metric points recorded"


# --------------------------------------------------------------------------- #
# end-to-end (extract + transform + load)
# --------------------------------------------------------------------------- #


@patch("tgedr_fdafaers.etl.compute_rows_metrics.Metrics")
@patch("tgedr_fdafaers.etl.compute_rows_metrics.load_dataset")
def test_full_etl_pipeline(mock_load_dataset, mock_metrics_cls) -> None:
    """Full pipeline should extract, transform, and record gauge metrics."""
    rows = [{"period": "24q1"}, {"period": "24q1"}, {"period": "24q2"}]
    mock_load_dataset.side_effect = lambda *args, **kwargs: _streaming_dataset(rows)

    mock_metrics = MagicMock()
    mock_metrics_cls.instance.return_value = mock_metrics

    etl = ComputeRowsMetrics(configuration={"dataset_prefix": "org/faers_bronze_"})
    etl.extract()
    etl.transform()
    result = etl.load()

    # one table present across all Constants.TABLES, each with the same two periods
    expected_points = len(Constants.TABLES) * 2
    assert mock_metrics.add_to_gauge.call_count == expected_points
    for table in Constants.TABLES:
        mock_metrics.add_to_gauge.assert_any_call(
            ComputeRowsMetrics.METRIC_NAME, 2, {"table": table, "period": "24q1"}
        )
        mock_metrics.add_to_gauge.assert_any_call(
            ComputeRowsMetrics.METRIC_NAME, 1, {"table": table, "period": "24q2"}
        )
    assert result == f"{expected_points} metric points recorded"

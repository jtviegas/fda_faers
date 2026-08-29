"""Unit tests for the ComputeRowsMetrics ETL."""

from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from tgedr_fdafaers.etl.compute_rows_metrics import ComputeRowsMetrics
from tgedr_fdafaers.constants import Constants


# --------------------------------------------------------------------------- #
# extract
# --------------------------------------------------------------------------- #


@patch("tgedr_fdafaers.etl.compute_rows_metrics.HuggingFaceDatasetStore")
def test_extract_loads_each_table_dataset(mock_store_cls) -> None:
    """extract should read each table's bronze dataset and store the train split."""
    mock_store = MagicMock()
    mock_store_cls.return_value = mock_store

    splits = MagicMock()
    splits.train = pd.DataFrame({"period": ["24q1", "24q1", "24q2"]})
    mock_store.get.return_value = splits

    etl = ComputeRowsMetrics(configuration={"dataset_prefix": "org/faers_bronze_"})
    etl.extract()

    assert mock_store.get.call_count == len(Constants.TABLES)
    for table in Constants.TABLES:
        assert table in etl._data


@patch("tgedr_fdafaers.etl.compute_rows_metrics.HuggingFaceDatasetStore")
def test_extract_raises_on_missing_dataset(mock_store_cls) -> None:
    """extract should raise EtlException when a required dataset is missing."""
    from tgedr_dataops.store.hf_dataset import NoStoreException
    from tgedr_dataops_abs.etl import EtlException

    mock_store = MagicMock()
    mock_store_cls.return_value = mock_store
    mock_store.get.side_effect = NoStoreException("not found")

    etl = ComputeRowsMetrics(configuration={"dataset_prefix": "org/faers_bronze_"})
    with pytest.raises(EtlException):
        etl.extract()

    assert etl._data == {}


@patch("tgedr_fdafaers.etl.compute_rows_metrics.HuggingFaceDatasetStore")
def test_extract_raises_on_empty_dataset(mock_store_cls) -> None:
    """extract should raise EtlException when a required dataset's train split is empty."""
    from tgedr_dataops_abs.etl import EtlException

    mock_store = MagicMock()
    mock_store_cls.return_value = mock_store

    empty_splits = MagicMock()
    empty_splits.train = pd.DataFrame()
    mock_store.get.return_value = empty_splits

    etl = ComputeRowsMetrics(configuration={"dataset_prefix": "org/faers_bronze_"})
    with pytest.raises(EtlException):
        etl.extract()

    assert etl._data == {}


@patch("tgedr_fdafaers.etl.compute_rows_metrics.HuggingFaceDatasetStore")
def test_extract_raises_on_none_train_split(mock_store_cls) -> None:
    """extract should raise EtlException when a required dataset's train split is None."""
    from tgedr_dataops_abs.etl import EtlException

    mock_store = MagicMock()
    mock_store_cls.return_value = mock_store

    none_splits = MagicMock()
    none_splits.train = None
    mock_store.get.return_value = none_splits

    etl = ComputeRowsMetrics(configuration={"dataset_prefix": "org/faers_bronze_"})
    with pytest.raises(EtlException):
        etl.extract()

    assert etl._data == {}


@patch("tgedr_fdafaers.etl.compute_rows_metrics.HuggingFaceDatasetStore")
def test_extract_uses_prefix_and_table_for_key(mock_store_cls) -> None:
    """extract should build dataset keys as '{prefix}{table}'."""
    mock_store = MagicMock()
    mock_store_cls.return_value = mock_store

    splits = MagicMock()
    splits.train = pd.DataFrame({"period": ["24q1"]})
    mock_store.get.return_value = splits

    etl = ComputeRowsMetrics(configuration={"dataset_prefix": "org/faers_"})
    etl.extract()

    expected_keys = [f"org/faers_{table}" for table in Constants.TABLES]
    actual_keys = [call.kwargs["key"] for call in mock_store.get.call_args_list]
    assert actual_keys == expected_keys


# --------------------------------------------------------------------------- #
# transform
# --------------------------------------------------------------------------- #


def test_transform_computes_rows_per_period() -> None:
    """transform should group by period and count rows per period for each table."""
    etl = ComputeRowsMetrics()
    etl._data = {
        "reac": pd.DataFrame({"period": ["24q1", "24q1", "24q2"]}),
        "drug": pd.DataFrame({"period": ["24q1", "24q2", "24q2", "24q2"]}),
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
    """transform should store period keys as strings even when groupby yields non-str."""
    etl = ComputeRowsMetrics()
    etl._data = {"reac": pd.DataFrame({"period": [241, 241, 242]})}

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
@patch("tgedr_fdafaers.etl.compute_rows_metrics.HuggingFaceDatasetStore")
def test_full_etl_pipeline(mock_store_cls, mock_metrics_cls) -> None:
    """Full pipeline should extract, transform, and record gauge metrics."""
    mock_store = MagicMock()
    mock_store_cls.return_value = mock_store

    splits = MagicMock()
    splits.train = pd.DataFrame({"period": ["24q1", "24q1", "24q2"]})
    mock_store.get.return_value = splits

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

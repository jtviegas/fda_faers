"""Unit tests for the IngestPeriodFiles ETL."""

from unittest.mock import patch, MagicMock, call

import pandas as pd
import pytest

from tgedr_fdafaers.etl.ingest_period_files import IngestPeriodFiles, _contract_path


# --------------------------------------------------------------------------- #
# extract
# --------------------------------------------------------------------------- #


@patch("tgedr_fdafaers.etl.ingest_period_files.RawDataIngestion")
@patch("tgedr_fdafaers.etl.ingest_period_files.pd.read_csv")
def test_extract_processes_each_file(mock_read_csv, mock_ingestion_cls, tmp_path) -> None:
    """extract should read and process each comma-separated file path."""
    mock_ingestion = MagicMock()
    mock_ingestion_cls.return_value = mock_ingestion

    df_reac = pd.DataFrame({"primaryid": [1], "period": ["24q1"]})
    mock_read_csv.return_value = df_reac
    mock_ingestion.process.return_value = df_reac

    etl = IngestPeriodFiles(configuration={"files": "/tmp/reac24q1.txt,/tmp/reac24q2.txt"})
    etl.extract()

    assert mock_read_csv.call_count == 2
    assert mock_ingestion.process.call_count == 2


@patch("tgedr_fdafaers.etl.ingest_period_files.RawDataIngestion")
@patch("tgedr_fdafaers.etl.ingest_period_files.pd.read_csv")
def test_extract_derives_table_from_filename(mock_read_csv, mock_ingestion_cls) -> None:
    """extract should use the first 4 chars of the filename as the table name."""
    mock_ingestion = MagicMock()
    mock_ingestion_cls.return_value = mock_ingestion
    mock_ingestion_cls.CONTEXT_KEY_TABLE = "table"
    mock_ingestion_cls.CONTEXT_KEY_DATAFRAME = "dataframe"

    df = pd.DataFrame({"primaryid": [1], "period": ["24q1"]})
    mock_read_csv.return_value = df
    mock_ingestion.process.return_value = df

    etl = IngestPeriodFiles(configuration={"files": "/data/reac24q1.txt"})
    etl.extract()

    # The table passed to process should be "reac" (first 4 chars of filename)
    context_arg = mock_ingestion.process.call_args.kwargs["context"]
    assert context_arg["table"] == "reac"


@patch("tgedr_fdafaers.etl.ingest_period_files.RawDataIngestion")
@patch("tgedr_fdafaers.etl.ingest_period_files.pd.read_csv")
def test_extract_concatenates_same_table_data(mock_read_csv, mock_ingestion_cls) -> None:
    """extract should concat DataFrames when multiple files map to the same table."""
    mock_ingestion = MagicMock()
    mock_ingestion_cls.return_value = mock_ingestion

    df1 = pd.DataFrame({"primaryid": [1], "period": ["24q1"]})
    df2 = pd.DataFrame({"primaryid": [2], "period": ["24q2"]})
    mock_read_csv.side_effect = [df1, df2]
    mock_ingestion.process.side_effect = [df1, df2]

    etl = IngestPeriodFiles(configuration={"files": "/data/reac24q1.txt,/data/reac24q2.txt"})
    etl.extract()

    assert "reac" in etl._data
    assert len(etl._data["reac"]) == 2


@patch("tgedr_fdafaers.etl.ingest_period_files.RawDataIngestion")
@patch("tgedr_fdafaers.etl.ingest_period_files.pd.read_csv")
def test_extract_uses_dollar_delimiter(mock_read_csv, mock_ingestion_cls) -> None:
    """extract should read CSV files with '$' as delimiter."""
    mock_ingestion = MagicMock()
    mock_ingestion_cls.return_value = mock_ingestion

    df = pd.DataFrame({"primaryid": [1], "period": ["24q1"]})
    mock_read_csv.return_value = df
    mock_ingestion.process.return_value = df

    etl = IngestPeriodFiles(configuration={"files": "/data/reac24q1.txt"})
    etl.extract()

    _, kwargs = mock_read_csv.call_args
    assert kwargs["delimiter"] == "$"


# --------------------------------------------------------------------------- #
# transform
# --------------------------------------------------------------------------- #


def test_transform_is_noop() -> None:
    """transform should not modify state."""
    etl = IngestPeriodFiles()
    etl._data = {"reac": pd.DataFrame({"primaryid": [1], "period": ["24q1"]})}

    etl.transform()

    assert len(etl._data["reac"]) == 1


# --------------------------------------------------------------------------- #
# load
# --------------------------------------------------------------------------- #


@patch("tgedr_fdafaers.etl.ingest_period_files.Metrics")
@patch("tgedr_fdafaers.etl.ingest_period_files.ContractedHFDatasetFileBasedStore")
def test_load_uploads_data_and_returns_periods(mock_store_cls, mock_metrics_cls) -> None:
    """load should upload each table's data and return unique periods sorted."""
    mock_store = MagicMock()
    mock_store_cls.return_value = mock_store
    mock_metrics = MagicMock()
    mock_metrics_cls.instance.return_value = mock_metrics

    etl = IngestPeriodFiles(configuration={"dataset_prefix": "org/faers"})
    etl._data = {
        "reac": pd.DataFrame({"primaryid": [1, 2], "period": ["24q2", "24q1"]}),
        "drug": pd.DataFrame({"primaryid": [3], "period": ["24q1"]}),
    }

    result = etl.load()

    assert result == "24q1,24q2"
    assert mock_store.save.call_count == 2


@patch("tgedr_fdafaers.etl.ingest_period_files.Metrics")
@patch("tgedr_fdafaers.etl.ingest_period_files.ContractedHFDatasetFileBasedStore")
def test_load_returns_empty_string_when_no_data(mock_store_cls, mock_metrics_cls) -> None:
    """load should return empty string when _data is empty."""
    mock_store = MagicMock()
    mock_store_cls.return_value = mock_store
    mock_metrics = MagicMock()
    mock_metrics_cls.instance.return_value = mock_metrics

    etl = IngestPeriodFiles(configuration={"dataset_prefix": "org/faers"})
    etl._data = {}

    result = etl.load()

    assert result == ""
    mock_store.save.assert_not_called()


@patch("tgedr_fdafaers.etl.ingest_period_files.Metrics")
@patch("tgedr_fdafaers.etl.ingest_period_files.ContractedHFDatasetFileBasedStore")
def test_load_uses_correct_dataset_name(mock_store_cls, mock_metrics_cls) -> None:
    """load should construct dataset names as '{prefix}{table}'."""
    mock_store = MagicMock()
    mock_store_cls.return_value = mock_store
    mock_metrics = MagicMock()
    mock_metrics_cls.instance.return_value = mock_metrics

    etl = IngestPeriodFiles(configuration={"dataset_prefix": "org/faers"})
    etl._data = {
        "reac": pd.DataFrame({"primaryid": [1], "period": ["24q1"]}),
    }

    etl.load()

    save_call = mock_store.save.call_args
    assert save_call.kwargs["key"] == "org/faersreac"


@patch("tgedr_fdafaers.etl.ingest_period_files.Metrics")
@patch("tgedr_fdafaers.etl.ingest_period_files.ContractedHFDatasetFileBasedStore")
def test_load_saves_with_train_split_and_append(mock_store_cls, mock_metrics_cls) -> None:
    """load should save each table with split='train' and append=True."""
    mock_store = MagicMock()
    mock_store_cls.return_value = mock_store
    mock_metrics = MagicMock()
    mock_metrics_cls.instance.return_value = mock_metrics

    etl = IngestPeriodFiles(configuration={"dataset_prefix": "org/faers"})
    etl._data = {
        "reac": pd.DataFrame({"primaryid": [1], "period": ["24q1"]}),
    }

    etl.load()

    save_call = mock_store.save.call_args
    assert save_call.kwargs["split"] == "train"
    assert save_call.kwargs["append"] is True
    assert save_call.kwargs["data_contract"] == _contract_path("reac")


@patch("tgedr_fdafaers.etl.ingest_period_files.Metrics")
@patch("tgedr_fdafaers.etl.ingest_period_files.ContractedHFDatasetFileBasedStore")
def test_load_passes_correct_contract_per_table(mock_store_cls, mock_metrics_cls) -> None:
    """load should pass each table's ODCS contract path to save."""
    mock_store = MagicMock()
    mock_store_cls.return_value = mock_store
    mock_metrics = MagicMock()
    mock_metrics_cls.instance.return_value = mock_metrics

    etl = IngestPeriodFiles(configuration={"dataset_prefix": "org/faers"})
    etl._data = {
        "reac": pd.DataFrame({"primaryid": [1], "period": ["24q1"]}),
        "drug": pd.DataFrame({"primaryid": [3], "period": ["24q1"]}),
    }

    etl.load()

    calls = mock_store.save.call_args_list
    contract_paths = {c.kwargs["key"]: c.kwargs["data_contract"] for c in calls}
    assert contract_paths == {
        "org/faersreac": _contract_path("reac"),
        "org/faersdrug": _contract_path("drug"),
    }


@patch("tgedr_fdafaers.etl.ingest_period_files.Metrics")
@patch("tgedr_fdafaers.etl.ingest_period_files.ContractedHFDatasetFileBasedStore")
def test_load_instantiates_store_with_public_visibility(mock_store_cls, mock_metrics_cls) -> None:
    """load should create the store with visibility=public."""
    mock_store = MagicMock()
    mock_store_cls.return_value = mock_store
    mock_metrics = MagicMock()
    mock_metrics_cls.instance.return_value = mock_metrics

    etl = IngestPeriodFiles(configuration={"dataset_prefix": "org/faers"})
    etl._data = {
        "reac": pd.DataFrame({"primaryid": [1], "period": ["24q1"]}),
    }

    etl.load()

    mock_store_cls.assert_called_once_with(config={"visibility": "public"})


@patch("tgedr_fdafaers.etl.ingest_period_files.Metrics")
@patch("tgedr_fdafaers.etl.ingest_period_files.ContractedHFDatasetFileBasedStore")
def test_load_records_rows_metric_per_table(mock_store_cls, mock_metrics_cls) -> None:
    """load should record a gauge metric with the number of rows per table."""
    mock_store = MagicMock()
    mock_store_cls.return_value = mock_store
    mock_metrics = MagicMock()
    mock_metrics_cls.instance.return_value = mock_metrics

    etl = IngestPeriodFiles(configuration={"dataset_prefix": "org/faers"})
    etl._data = {
        "reac": pd.DataFrame({"primaryid": [1, 2], "period": ["24q1", "24q1"]}),
    }

    etl.load()

    mock_metrics.add_to_gauge.assert_called_once_with(
        "fda_faers.ingest_period_files.new_rows", 2, {"table": "reac"}
    )


# --------------------------------------------------------------------------- #
# end-to-end
# --------------------------------------------------------------------------- #


@patch("tgedr_fdafaers.etl.ingest_period_files.Metrics")
@patch("tgedr_fdafaers.etl.ingest_period_files.ContractedHFDatasetFileBasedStore")
@patch("tgedr_fdafaers.etl.ingest_period_files.RawDataIngestion")
@patch("tgedr_fdafaers.etl.ingest_period_files.pd.read_csv")
def test_full_etl_pipeline(mock_read_csv, mock_ingestion_cls, mock_store_cls, mock_metrics_cls) -> None:
    """Full pipeline should extract, transform (noop), and load data."""
    mock_ingestion = MagicMock()
    mock_ingestion_cls.return_value = mock_ingestion
    mock_store = MagicMock()
    mock_store_cls.return_value = mock_store
    mock_metrics = MagicMock()
    mock_metrics_cls.instance.return_value = mock_metrics

    df = pd.DataFrame({"primaryid": [1], "period": ["24q1"]})
    mock_read_csv.return_value = df
    mock_ingestion.process.return_value = df

    etl = IngestPeriodFiles(configuration={"files": "/data/reac24q1.txt", "dataset_prefix": "org/faers"})
    etl.extract()
    etl.transform()
    result = etl.load()

    assert result == "24q1"
    mock_store.save.assert_called_once()

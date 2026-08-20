"""Unit tests for the RawDataIngestion processor."""

import numpy as np
import pandas as pd
import pytest

from tgedr_fdafaers.raw_data_ingestion import RawDataIngestion


# --------------------------------------------------------------------------- #
# get_schema
# --------------------------------------------------------------------------- #


def test_get_schema_returns_dict_for_known_table() -> None:
    """get_schema should return a column->dtype dict for a known table."""
    processor = RawDataIngestion()

    schema = processor.get_schema("reac")

    assert isinstance(schema, dict)
    assert "primaryid" in schema
    assert schema["primaryid"] == "Int64"
    assert schema["pt"] == "string"


def test_get_schema_raises_for_unknown_table() -> None:
    """get_schema should raise ValueError for an unrecognized table name."""
    processor = RawDataIngestion()

    with pytest.raises(ValueError, match="No predefined schema"):
        processor.get_schema("unknown_table")


# --------------------------------------------------------------------------- #
# process - context validation
# --------------------------------------------------------------------------- #


def test_process_raises_when_context_is_none() -> None:
    """process should raise when context is None."""
    processor = RawDataIngestion()

    with pytest.raises(Exception, match="you must provide"):
        processor.process(None)


def test_process_raises_when_context_missing_table() -> None:
    """process should raise when 'table' key is missing."""
    processor = RawDataIngestion()
    df = pd.DataFrame({"primaryid": [1]})

    with pytest.raises(Exception, match="you must provide"):
        processor.process({"dataframe": df})


def test_process_raises_when_context_missing_dataframe() -> None:
    """process should raise when 'dataframe' key is missing."""
    processor = RawDataIngestion()

    with pytest.raises(Exception, match="you must provide"):
        processor.process({"table": "reac"})


# --------------------------------------------------------------------------- #
# process - full pipeline
# --------------------------------------------------------------------------- #


def _make_reac_df(**overrides) -> pd.DataFrame:
    """Create a minimal 'reac' DataFrame with defaults."""
    data = {
        "primaryid": [1],
        "caseid": [100],
        "pt": ["HEADACHE"],
        "drug_rec_act": ["action"],
        "processing_time": [20240101],
        "period": ["24q1"],
    }
    data.update(overrides)
    return pd.DataFrame(data)


def test_process_reac_table_returns_correct_columns() -> None:
    """process should return a DataFrame with schema-ordered columns for 'reac'."""
    processor = RawDataIngestion()
    df = _make_reac_df()

    result = processor.process({"table": "reac", "dataframe": df})

    expected_cols = ["primaryid", "caseid", "pt", "drug_rec_act", "processing_time", "period"]
    assert list(result.columns) == expected_cols


def test_process_casts_int_columns_to_int64() -> None:
    """process should cast integer columns to nullable Int64."""
    processor = RawDataIngestion()
    df = _make_reac_df(primaryid=["123"], caseid=["456"])

    result = processor.process({"table": "reac", "dataframe": df})

    assert result["primaryid"].dtype == pd.Int64Dtype()
    assert result["caseid"].dtype == pd.Int64Dtype()
    assert result["primaryid"].iloc[0] == 123


def test_process_casts_string_columns_to_string_dtype() -> None:
    """process should cast string columns to pandas StringDtype."""
    processor = RawDataIngestion()
    df = _make_reac_df()

    result = processor.process({"table": "reac", "dataframe": df})

    assert result["pt"].dtype == pd.StringDtype()


def test_process_handles_nan_sentinel_in_string_columns() -> None:
    """process should replace 'None' and 'nan' strings with pd.NA in string columns."""
    processor = RawDataIngestion()
    df = _make_reac_df(pt=["None"], drug_rec_act=["nan"])

    result = processor.process({"table": "reac", "dataframe": df})

    assert pd.isna(result["pt"].iloc[0])
    assert pd.isna(result["drug_rec_act"].iloc[0])


def test_process_coerces_invalid_numeric_to_na() -> None:
    """process should coerce non-numeric strings in Int64 columns to pd.NA."""
    processor = RawDataIngestion()
    df = _make_reac_df(primaryid=["abc"])

    result = processor.process({"table": "reac", "dataframe": df})

    assert pd.isna(result["primaryid"].iloc[0])


def test_process_drops_rescued_data_column() -> None:
    """process should drop the '_rescued_data' column if present."""
    processor = RawDataIngestion()
    df = _make_reac_df()
    df["_rescued_data"] = ["some junk"]

    result = processor.process({"table": "reac", "dataframe": df})

    assert "_rescued_data" not in result.columns


def test_process_raises_on_schema_mismatch() -> None:
    """process should raise ValueError when columns don't match the expected schema."""
    processor = RawDataIngestion()
    # Missing required columns
    df = pd.DataFrame({"primaryid": [1], "unknown_col": ["x"]})

    with pytest.raises(ValueError, match="Schema mismatch"):
        processor.process({"table": "reac", "dataframe": df})


# --------------------------------------------------------------------------- #
# _sort_columns - unknown table
# --------------------------------------------------------------------------- #


def test_sort_columns_no_op_for_unknown_table() -> None:
    """_sort_columns should not reorder when the table has no schema."""
    processor = RawDataIngestion()
    df = pd.DataFrame({"b": [1], "a": [2]})

    result = processor._sort_columns(df, "nonexistent_table")

    assert list(result.columns) == ["b", "a"]


# --------------------------------------------------------------------------- #
# process with Float64 columns
# --------------------------------------------------------------------------- #


def test_process_casts_float_columns() -> None:
    """process should cast Float64 columns correctly."""
    processor = RawDataIngestion()
    schema = processor.get_schema("ther")
    # Build a minimal 'ther' DataFrame
    df = pd.DataFrame({
        "primaryid": [1],
        "caseid": [100],
        "dsg_drug_seq": [1],
        "start_dt": [20240101],
        "end_dt": [20240201],
        "dur": ["3.5"],
        "dur_cod": ["DAY"],
        "processing_time": [20240101],
        "period": ["24q1"],
    })

    result = processor.process({"table": "ther", "dataframe": df})

    assert result["dur"].dtype == pd.Float64Dtype()
    assert result["dur"].iloc[0] == 3.5

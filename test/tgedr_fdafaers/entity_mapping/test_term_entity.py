"""Unit tests for the TermEntity processor."""

from unittest.mock import patch

import pandas as pd
import pytest

from tgedr_dataops_abs.processor import ProcessorException

from tgedr_fdafaers.entity_mapping import term_entity as term_entity_module
from tgedr_fdafaers.entity_mapping.term_entity import TermEntity


def _term_df() -> pd.DataFrame:
    return pd.DataFrame({"term": ["aspirin", "insulin novulin 70 30", "unknown"]})


def test_process_raises_when_context_is_none() -> None:
    """process should raise ProcessorException when context is None."""
    with pytest.raises(ProcessorException, match="dataframe must be provided"):
        TermEntity().process(context=None)


def test_process_raises_when_dataframe_missing() -> None:
    """process should raise ProcessorException when the dataframe key is missing."""
    with pytest.raises(ProcessorException, match="dataframe must be provided"):
        TermEntity().process(context={})


@patch.object(term_entity_module, "parse_entities")
def test_process_explodes_entities_into_rows(mock_parse_entities) -> None:
    """process should explode the parsed entities into individual rows."""

    def fake_parse(text):  # noqa: ANN001, ANN202
        return {
            "aspirin": ["aspirin"],
            "insulin novulin 70 30": ["insulin", "insulin novulin 70 30"],
            "unknown": ["unknown"],
        }[text]

    mock_module = term_entity_module
    with patch.object(mock_module, "parse_entities", fake_parse):
        result = TermEntity().process(context={"dataframe": _term_df()})

    assert result["entity"].tolist() == ["aspirin", "insulin", "insulin novulin 70 30", "unknown"]


@patch.object(term_entity_module, "parse_entities")
def test_process_dedupes_fully_duplicate_rows(mock_parse_entities) -> None:
    """process should drop rows duplicated across all columns."""

    def fake_parse(text):  # noqa: ANN001, ANN202
        if text == "aspirin":
            return ["aspirin"]
        return ["aspirin"]

    with patch.object(term_entity_module, "parse_entities", fake_parse):
        df = pd.DataFrame({"term": ["aspirin", "aspirin"]})
        result = TermEntity().process(context={"dataframe": df})

    assert result["entity"].tolist() == ["aspirin"]


@patch.object(term_entity_module, "parse_entities")
def test_process_uses_custom_column_names(mock_parse_entities) -> None:
    """process should use the custom term and entity column names."""

    def fake_parse(text):  # noqa: ANN001, ANN202
        return [text]

    with patch.object(term_entity_module, "parse_entities", fake_parse):
        df = pd.DataFrame({"drug": ["aspirin"]})
        result = TermEntity().process(
            context={"dataframe": df, "term_column": "drug", "entity_column": "result"}
        )

    assert "result" in result.columns
    assert result["result"].tolist() == ["aspirin"]

"""Unit tests for the DrugIngredientTerm processor."""

import pandas as pd
import pytest

from tgedr_dataops_abs.processor import ProcessorException

from tgedr_fdafaers.entity_mapping.drug_ingredient_term import DrugIngredientTerm


def _drug_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "primaryid": [1, 2, 3],
            "period": ["24q1", "24q1", "24q1"],
            "drugname": ["PROPOFAN", "ASPIRIN", "INSULIN NOVULIN 70/30"],
            "prod_ai": [None, "ACETYLSALICYLIC ACID", None],
        }
    )


def test_process_raises_when_context_is_none() -> None:
    """process should raise ProcessorException when context is None."""
    with pytest.raises(ProcessorException, match="dataframe must be provided"):
        DrugIngredientTerm().process(context=None)


def test_process_raises_when_dataframe_missing() -> None:
    """process should raise ProcessorException when the dataframe key is missing."""
    with pytest.raises(ProcessorException, match="dataframe must be provided"):
        DrugIngredientTerm().process(context={})


def test_process_adds_term_column() -> None:
    """process should add a normalized term column to the dataframe."""
    result = DrugIngredientTerm().process(context={"dataframe": _drug_df()})

    assert "term" in result.columns
    assert result["term"].tolist() == ["propofan", "acetylsalicylic acid", "insulin novulin 70 30"]


def test_process_prefers_ingredient_over_drugname() -> None:
    """process should use prod_ai when available, otherwise drugname."""
    result = DrugIngredientTerm().process(context={"dataframe": _drug_df()})

    row = result[result["primaryid"] == 2].iloc[0]
    assert row["term"] == "acetylsalicylic acid"


def test_process_deduplicates_terms() -> None:
    """process should drop fully duplicate rows."""
    df = pd.DataFrame(
        {
            "primaryid": [1, 1],
            "period": ["24q1", "24q1"],
            "drugname": ["PROPOFAN", "PROPOFAN"],
            "prod_ai": [None, None],
        }
    )
    result = DrugIngredientTerm().process(context={"dataframe": df})

    assert len(result) == 1


def test_process_uses_custom_term_column_name() -> None:
    """process should rename the term column when term_column is provided."""
    result = DrugIngredientTerm().process(
        context={"dataframe": _drug_df(), "term_column": "entity"}
    )

    assert "entity" in result.columns
    assert "term" not in result.columns
    assert result["entity"].tolist() == ["propofan", "acetylsalicylic acid", "insulin novulin 70 30"]

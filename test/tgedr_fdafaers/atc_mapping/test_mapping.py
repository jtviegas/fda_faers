"""Unit tests for the ATC mapping base module."""

import pandas as pd
import pytest

from tgedr_fdafaers.atc_mapping.mapping import AtcClass, AtcMapping


# --------------------------------------------------------------------------- #
# AtcClass
# --------------------------------------------------------------------------- #


def test_atc_class_to_dict() -> None:
    """to_dict should return the expected dictionary."""
    atc_class = AtcClass(id="A01", name="X", class_type="ATC")

    assert atc_class.to_dict() == {"id": "A01", "name": "X", "class_type": "ATC"}


def test_atc_class_from_dict() -> None:
    """from_dict should reconstruct an AtcClass instance."""
    atc_class = AtcClass.from_dict({"id": "A01", "name": "X", "class_type": "ATC"})

    assert isinstance(atc_class, AtcClass)
    assert atc_class.id == "A01"
    assert atc_class.name == "X"
    assert atc_class.class_type == "ATC"


def test_atc_class_is_frozen() -> None:
    """AtcClass instances should be immutable."""
    atc_class = AtcClass(id="A01", name="X", class_type="ATC")

    with pytest.raises(Exception):
        atc_class.id = "B02"  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# AtcMapping
# --------------------------------------------------------------------------- #


class _ConcreteAtcMapping(AtcMapping):
    """Concrete subclass so the abstract base class can be exercised."""

    def decorate(self, df: pd.DataFrame) -> pd.DataFrame:
        return super().decorate(df)


def test_atc_mapping_schema() -> None:
    """ATC_CODE_SCHEMA should describe the three string columns."""
    assert AtcMapping.ATC_CODE_SCHEMA == {
        "id": "string",
        "name": "string",
        "class_type": "string",
    }


def test_atc_mapping_init_defaults() -> None:
    """Default column names should be term/atc_codes."""
    mapping = _ConcreteAtcMapping()

    assert mapping._term_col == "term"
    assert mapping._output_col == "atc_codes"


def test_atc_mapping_init_custom_columns() -> None:
    """Custom column names should be honoured."""
    mapping = _ConcreteAtcMapping(term_col="drug", output_col="codes")

    assert mapping._term_col == "drug"
    assert mapping._output_col == "codes"


def test_atc_mapping_decorate_is_abstract() -> None:
    """The abstract decorate method should raise NotImplementedError."""
    mapping = _ConcreteAtcMapping()

    with pytest.raises(NotImplementedError):
        mapping.decorate(pd.DataFrame({"term": ["aspirin"]}))

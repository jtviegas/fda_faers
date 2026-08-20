"""Unit tests for the Constants frozen dataclass."""

from dataclasses import FrozenInstanceError

import pytest

from tgedr_fdafaers.constants import Constants


def test_csv_delimiter_value() -> None:
    """CSV_DELIMITER should be the dollar sign."""
    c = Constants()

    assert c.CSV_DELIMITER == "$"


def test_tables_contains_expected_entries() -> None:
    """TABLES should list all seven FAERS table names."""
    c = Constants()

    expected = ("demo", "drug", "indi", "outc", "reac", "rpsr", "ther")
    assert c.TABLES == expected


def test_tables_length() -> None:
    """TABLES tuple should have exactly 7 entries."""
    c = Constants()

    assert len(c.TABLES) == 7


def test_frozen_prevents_attribute_assignment() -> None:
    """Constants should be immutable (frozen dataclass)."""
    c = Constants()

    with pytest.raises(FrozenInstanceError):
        c.CSV_DELIMITER = "|"


def test_frozen_prevents_tables_reassignment() -> None:
    """Reassigning TABLES should raise FrozenInstanceError."""
    c = Constants()

    with pytest.raises(FrozenInstanceError):
        c.TABLES = ("other",)


def test_default_instance_matches_class_defaults() -> None:
    """Creating a Constants instance without arguments should use defaults."""
    c = Constants()

    assert isinstance(c.CSV_DELIMITER, str)
    assert isinstance(c.TABLES, tuple)
    assert all(isinstance(t, str) for t in c.TABLES)

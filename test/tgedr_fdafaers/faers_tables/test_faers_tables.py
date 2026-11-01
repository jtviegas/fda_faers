"""Tests for the standardized FAERS table bundle."""

import pytest
from datasets import Dataset

from tgedr_fdafaers.faers_tables.faers_tables import FaersTables


def _tables() -> FaersTables:
    empty = Dataset.from_list([])
    return FaersTables(demo=empty, drug=empty, reac=empty, outc=empty, rpsr=empty, ther=empty, indi=empty)


def test_faers_tables_dict_round_trip() -> None:
    """Convert table bundles to and from table-name keyed dictionaries."""
    tables = _tables()

    assert FaersTables.from_dict(tables.to_dict()) == tables


def test_faers_tables_requires_all_seven_tables() -> None:
    """Reject a mapping that omits a required table."""
    tables = _tables().to_dict()

    with pytest.raises(ValueError, match="Missing FAERS tables: indi"):
        FaersTables.from_dict({name: dataset for name, dataset in tables.items() if name != "indi"})

"""Tests for shared FAERS processor helpers."""

import pytest
from datasets import Dataset
from tgedr_dataops_abs.processor import ProcessorException

from tgedr_fdafaers.faers_processor import FaersProcessor


class _TestProcessor(FaersProcessor):
    def process(self, context=None):
        return context


@pytest.fixture
def processor() -> FaersProcessor:
    """Return a concrete processor instance for helper tests."""
    return _TestProcessor()


def test_require_columns_accepts_present_columns(processor: FaersProcessor) -> None:
    """Accept a dataset containing all requested columns."""
    processor._require_columns(Dataset.from_list([{"primaryid": 7, "caseid": 2}]), "demo", {"primaryid", "caseid"})


def test_require_columns_reports_missing_columns(processor: FaersProcessor) -> None:
    """Raise a processor exception naming the missing dataset columns."""
    with pytest.raises(ProcessorException, match="demo is missing required columns: caseid, primaryid"):
        processor._require_columns(Dataset.from_list([{"age": 40}]), "demo", {"primaryid", "caseid"})


def test_version_key_orders_missing_versions_and_integer_versions(processor: FaersProcessor) -> None:
    """Treat missing versions as older than numeric versions and compare numerically."""
    assert processor._version_key(None) == (0, -1)
    assert processor._version_key("2") < processor._version_key(10)


def test_unique_values_preserves_first_occurrences_and_omits_nulls(processor: FaersProcessor) -> None:
    """Return unique non-null row values in their original order."""
    rows = [{"pt": "RASH"}, {"pt": None}, {"pt": "RASH"}, {"pt": "NAUSEA"}, {}]

    assert processor._unique_values(rows, "pt") == ["RASH", "NAUSEA"]

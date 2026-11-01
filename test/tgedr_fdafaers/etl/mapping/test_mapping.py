"""Unit tests for the Mapping ETL."""

# ruff: noqa: SLF001

from unittest.mock import MagicMock, patch

import pytest
from datasets import Dataset
from tgedr_dataops.store.hf_dataset_file_arrow_based import NoStoreException

from tgedr_fdafaers.etl.mapping.mapping import Mapping


@pytest.fixture
def etl() -> Mapping:
    """Return a Mapping instance with a mocked store."""
    with patch("tgedr_fdafaers.etl.mapping.mapping.HuggingFaceDatasetFileArrowBasedStore") as mock_store_cls:
        mock_store = MagicMock()
        mock_store_cls.return_value = mock_store
        etl = Mapping(
            configuration={
                "bronze_dataset_prefix": "b/",
                "silver_dataset_prefix": "s/",
            }
        )
        etl._store = mock_store
        yield etl


def _dataset(rows: list[dict]) -> Dataset:
    """Build a Dataset from a list of row dicts."""
    return Dataset.from_list(rows)


# --------------------------------------------------------------------------- #
# extract
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("silver_missing", [False, True])
def test_extract_reads_unprocessed_bronze_drug_rows(etl: Mapping, silver_missing: bool) -> None:
    """Extract bronze drug rows whether or not the silver drug dataset exists."""
    stream = MagicMock()
    stream.filter.return_value.take.return_value = [{"primaryid": 2, "term": "aspirin"}]
    bronze = MagicMock(train=stream)
    if silver_missing:
        etl._store.get.side_effect = [NoStoreException("missing silver drug dataset"), bronze]
    else:
        etl._store.get.side_effect = [MagicMock(train=MagicMock(unique=lambda column: [1])), bronze]

    etl.extract()

    etl._store.get.assert_any_call("b/drug", streaming=True)
    assert etl._data["drug"][0]["term"] == "aspirin"


# --------------------------------------------------------------------------- #
# transform
# --------------------------------------------------------------------------- #


def test_transform_maps_known_terms_and_preserves_unmapped_rows(etl: Mapping) -> None:
    """Enrich terms found in term_atc and retain other rows with null codes."""
    etl._data["drug"] = _dataset([{"term": "aspirin"}, {"term": "unknown"}])
    etl._store.get.return_value = MagicMock(
        train=_dataset([{"term": "aspirin", "atc_codes": [{"id": "A01A"}]}])
    )

    etl.transform()

    result = list(etl._result["drug"])
    assert result == [
        {"term": "aspirin", "atc_codes": [{"id": "A01A"}]},
        {"term": "unknown", "atc_codes": None},
    ]
    etl._store.get.assert_called_once_with("s/term_atc")


def test_transform_handles_missing_term_atc_dataset(etl: Mapping) -> None:
    """Preserve all source terms with null ATC codes if the mapping dataset is absent."""
    from tgedr_dataops.store.hf_dataset_file_arrow_based import NoStoreException

    etl._data["drug"] = _dataset([{"term": "aspirin"}, {"term": "unknown"}])
    etl._store.get.side_effect = NoStoreException("missing mapping dataset")

    etl.transform()

    assert [row["atc_codes"] for row in etl._result["drug"]] == [None, None]


def test_transform_handles_empty_input_dataset(etl: Mapping) -> None:
    """Return an empty dataset with the added ATC-code column for empty input."""
    etl._data["drug"] = _dataset([])
    etl._store.get.return_value = MagicMock(train=_dataset([]))

    etl.transform()

    result = etl._result["drug"]
    assert result.num_rows == 0
    assert result.column_names == ["atc_codes"]


def test_transform_preserves_duplicate_mapping_codes(etl: Mapping) -> None:
    """Attach the ATC-code list stored for the matching term without altering it."""
    codes = [{"id": "A01A"}, {"id": "B01A"}]
    etl._data["drug"] = _dataset([{"term": "aspirin"}])
    etl._store.get.return_value = MagicMock(train=_dataset([{"term": "aspirin", "atc_codes": codes}]))

    etl.transform()

    assert etl._result["drug"][0]["atc_codes"] == codes


# --------------------------------------------------------------------------- #
# load
# --------------------------------------------------------------------------- #


def test_load_saves_result(etl: Mapping) -> None:
    """load should save each result to silver."""
    etl._result["drug"] = _dataset([{"term": "aspirin", "atc_codes": [{"id": "A01A"}]}])

    result = etl.load()

    etl._store.save.assert_called_once_with(
        df=etl._result["drug"],
        key="s/drug",
        split="train",
        append=True,
    )
    assert result == "s/drug"


def test_load_returns_empty_when_no_results(etl: Mapping) -> None:
    """load should return an empty string when there are no results."""
    etl._result = {}

    result = etl.load()

    assert result == ""
    etl._store.save.assert_not_called()

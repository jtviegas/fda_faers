"""Unit tests for the SilverLayer ETL."""

# ruff: noqa: SLF001

from unittest.mock import MagicMock, patch

import pytest
from datasets import Dataset

from tgedr_fdafaers.etl.silver_layer import SilverLayer


@pytest.fixture
def etl() -> SilverLayer:
    """Return a SilverLayer instance with a mocked store."""
    with patch("tgedr_fdafaers.etl.silver_layer.HuggingFaceDatasetFileArrowBasedStore") as mock_store_cls:
        mock_store = MagicMock()
        mock_store_cls.return_value = mock_store
        etl = SilverLayer(
            configuration={
                "bronze_dataset_prefix": "b/",
                "silver_dataset_prefix": "s/",
            }
        )
        etl._store = mock_store
        yield etl


@pytest.fixture
def mock_metrics() -> MagicMock:
    """Return a mocked Metrics.instance()."""
    with patch("tgedr_fdafaers.etl.silver_layer.Metrics") as mock_metrics_cls:
        mock_metrics = MagicMock()
        mock_metrics_cls.instance.return_value = mock_metrics
        yield mock_metrics


def _dataset(rows: list[dict]) -> Dataset:
    """Build a Dataset from a list of row dicts."""
    return Dataset.from_list(rows)


# --------------------------------------------------------------------------- #
# extract
# --------------------------------------------------------------------------- #


def test_extract_reads_bronze_tables(etl: SilverLayer) -> None:
    """extract should read the bronze drug table into _data."""
    mock_ds = MagicMock()
    mock_ds.train = _dataset([{"term": "aspirin"}])
    etl._store.get.return_value = mock_ds

    etl.extract()

    assert etl._store.get.call_args.args[0] == "b/drug"
    assert "drug" in etl._data
    assert etl._data["drug"][0]["term"] == "aspirin"


# --------------------------------------------------------------------------- #
# transform
# --------------------------------------------------------------------------- #


def test_transform_maps_known_terms_and_preserves_unmapped_rows(etl: SilverLayer) -> None:
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


def test_transform_records_total_mapped_and_unmapped_metrics(
    etl: SilverLayer, mock_metrics: MagicMock
) -> None:
    """Record row-count metrics for the input, mapped and unmapped rows."""
    etl._data["drug"] = _dataset([{"term": "aspirin"}, {"term": "unknown"}])
    etl._store.get.return_value = MagicMock(
        train=_dataset([{"term": "aspirin", "atc_codes": [{"id": "A01A"}]}])
    )

    etl.transform()

    assert mock_metrics.add_to_gauge.call_count == 3
    mock_metrics.add_to_gauge.assert_any_call(
        name="fda_faers.silver_layer.rows", value=2, attributes={"type": "silver_term"}
    )
    mock_metrics.add_to_gauge.assert_any_call(
        name="fda_faers.silver_layer.rows", value=1, attributes={"type": "silver_term_mapped"}
    )
    mock_metrics.add_to_gauge.assert_any_call(
        name="fda_faers.silver_layer.rows", value=1, attributes={"type": "silver_term_unmapped"}
    )


def test_transform_handles_missing_term_atc_dataset(etl: SilverLayer) -> None:
    """Preserve all source terms with null ATC codes if the mapping dataset is absent."""
    from tgedr_dataops.store.hf_dataset_file_arrow_based import NoStoreException

    etl._data["drug"] = _dataset([{"term": "aspirin"}, {"term": "unknown"}])
    etl._store.get.side_effect = NoStoreException("missing mapping dataset")

    etl.transform()

    assert [row["atc_codes"] for row in etl._result["drug"]] == [None, None]


def test_transform_handles_empty_input_dataset(etl: SilverLayer) -> None:
    """Return an empty dataset with the added ATC-code column for empty input."""
    etl._data["drug"] = _dataset([])
    etl._store.get.return_value = MagicMock(train=_dataset([]))

    etl.transform()

    result = etl._result["drug"]
    assert result.num_rows == 0
    assert result.column_names == ["atc_codes"]


def test_transform_preserves_duplicate_mapping_codes(etl: SilverLayer) -> None:
    """Attach the ATC-code list stored for the matching term without altering it."""
    codes = [{"id": "A01A"}, {"id": "B01A"}]
    etl._data["drug"] = _dataset([{"term": "aspirin"}])
    etl._store.get.return_value = MagicMock(train=_dataset([{"term": "aspirin", "atc_codes": codes}]))

    etl.transform()

    assert etl._result["drug"][0]["atc_codes"] == codes


# --------------------------------------------------------------------------- #
# load
# --------------------------------------------------------------------------- #


def test_load_saves_result(etl: SilverLayer) -> None:
    """load should save each result to silver."""
    etl._result["drug"] = _dataset([{"term": "aspirin", "atc_codes": [{"id": "A01A"}]}])

    result = etl.load()

    etl._store.save.assert_called_once_with(
        df=etl._result["drug"],
        key="s/drug",
        split="train",
    )
    assert result == "s/drug"


def test_load_returns_empty_when_no_results(etl: SilverLayer) -> None:
    """load should return an empty string when there are no results."""
    etl._result = {}

    result = etl.load()

    assert result == ""
    etl._store.save.assert_not_called()

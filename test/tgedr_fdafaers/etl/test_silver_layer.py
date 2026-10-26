"""Unit tests for the SilverLayer ETL."""

from unittest.mock import MagicMock, patch

import pytest
from datasets import Dataset

from tgedr_fdafaers.etl.silver_layer import SilverLayer


@pytest.fixture()
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


@pytest.fixture()
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


def test_transform_fills_missing_atc_codes(etl: SilverLayer, mock_metrics: MagicMock) -> None:
    """transform should fill atc_codes from the term->atc map when missing."""
    etl._data["drug"] = _dataset([{"term": "aspirin"}, {"term": "unknown"}])

    etl._store.get.side_effect = [
        MagicMock(train=_dataset([{"term": "aspirin", "entity": "aspirin"}])),
        MagicMock(train=_dataset([{"entity": "aspirin", "atc_codes": [{"id": "A01A"}]}])),
    ]

    etl.transform()

    result = etl._result["drug"]
    assert result[0]["atc_codes"] == [{"id": "A01A"}]
    assert result[1]["atc_codes"] == []


def test_transform_records_mapped_rows_metric(etl: SilverLayer, mock_metrics: MagicMock) -> None:
    """transform should record the input row count and the mapped row count as metrics."""
    etl._data["drug"] = _dataset([{"term": "aspirin"}, {"term": "unknown"}])

    etl._store.get.side_effect = [
        MagicMock(train=_dataset([{"term": "aspirin", "entity": "aspirin"}])),
        MagicMock(train=_dataset([{"entity": "aspirin", "atc_codes": [{"id": "A01A"}]}])),
    ]

    etl.transform()

    mock_metrics.add_to_gauge.assert_any_call(
        name="fda_faers.silver_layer.rows",
        value=2,
        attributes={"type": "drug"},
    )
    mock_metrics.add_to_gauge.assert_any_call(
        name="fda_faers.silver_layer.rows",
        value=1,
        attributes={"type": "drug_mapped"},
    )
    assert mock_metrics.add_to_gauge.call_count == 2


def test_transform_keeps_existing_atc_codes(etl: SilverLayer) -> None:
    """transform should not overwrite rows that already have atc_codes."""
    etl._data["drug"] = _dataset([{"term": "aspirin", "atc_codes": [{"id": "B01A"}]}])

    etl._store.get.side_effect = [
        MagicMock(train=_dataset([{"term": "aspirin", "entity": "aspirin"}])),
        MagicMock(train=_dataset([{"entity": "aspirin", "atc_codes": [{"id": "A01A"}]}])),
    ]

    etl.transform()

    assert etl._result["drug"][0]["atc_codes"] == [{"id": "B01A"}]


def test_transform_fills_none_atc_codes(etl: SilverLayer) -> None:
    """transform should treat None atc_codes as missing and fill from the map."""
    etl._data["drug"] = _dataset([{"term": "aspirin", "atc_codes": None}])

    etl._store.get.side_effect = [
        MagicMock(train=_dataset([{"term": "aspirin", "entity": "aspirin"}])),
        MagicMock(train=_dataset([{"entity": "aspirin", "atc_codes": [{"id": "A01A"}]}])),
    ]

    etl.transform()

    assert etl._result["drug"][0]["atc_codes"] == [{"id": "A01A"}]


def test_transform_aggregates_multiple_entities_and_atc(etl: SilverLayer) -> None:
    """transform should aggregate atc codes across all entities of a term."""
    etl._data["drug"] = _dataset([{"term": "aspirin"}])

    etl._store.get.side_effect = [
        MagicMock(
            train=_dataset(
                [
                    {"term": "aspirin", "entity": "aspirin"},
                    {"term": "aspirin", "entity": "acetylsalicylic"},
                ]
            )
        ),
        MagicMock(
            train=_dataset(
                [
                    {"entity": "aspirin", "atc_codes": [{"id": "A01A"}]},
                    {"entity": "acetylsalicylic", "atc_codes": [{"id": "B01A"}]},
                ]
            )
        ),
    ]

    etl.transform()

    assert etl._result["drug"][0]["atc_codes"] == [{"id": "A01A"}, {"id": "B01A"}]


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

"""Unit tests for the TermEntityMapping ETL."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from tgedr_dataops.store.contracted_store import NoStoreException

from tgedr_fdafaers.etl.term_entity_mapping import TermEntityMapping


@pytest.fixture()
def etl() -> TermEntityMapping:
    """Return a TermEntityMapping instance with a mocked store."""
    with patch("tgedr_fdafaers.etl.term_entity_mapping.ContractedHFDatasetFileBasedStore") as mock_store_cls:
        mock_store = MagicMock()
        mock_store_cls.return_value = mock_store
        etl = TermEntityMapping(
            configuration={
                "bronze_dataset_prefix": "b/",
                "silver_dataset_prefix": "s/",
                "sample_size": 10,
            }
        )
        etl._store = mock_store
        yield etl


@pytest.fixture()
def mock_metrics() -> MagicMock:
    """Return a mocked Metrics.instance()."""
    with patch("tgedr_fdafaers.etl.term_entity_mapping.Metrics") as mock_metrics_cls:
        mock_metrics = MagicMock()
        mock_metrics_cls.instance.return_value = mock_metrics
        yield mock_metrics


# --------------------------------------------------------------------------- #
# extract
# --------------------------------------------------------------------------- #


def test_extract_reads_bronze_terms(etl: TermEntityMapping, mock_metrics: MagicMock) -> None:
    """extract should read bronze drug terms and store a sample of them."""
    df_terms = pd.DataFrame({"term": ["a", "b", "c"]})
    etl._store.get.side_effect = [MagicMock(train=df_terms), NoStoreException("missing")]

    etl.extract()

    assert etl._store.get.call_args_list[0].kwargs["key"] == "b/drug"
    assert etl._store.get.call_args_list[1].kwargs["key"] == "s/term_entity"
    assert set(etl._data["terms_to_map"]["term"]) == {"a", "b", "c"}
    mock_metrics.add_to_gauge.assert_called()


def test_extract_excludes_terms_already_mapped(etl: TermEntityMapping, mock_metrics: MagicMock) -> None:
    """extract should exclude terms already present in the silver term_entity table."""
    df_terms = pd.DataFrame({"term": ["a", "b", "c", "d"]})
    df_mapped = pd.DataFrame({"term": ["a", "b"]})
    etl._store.get.side_effect = [MagicMock(train=df_terms), MagicMock(train=df_mapped)]

    etl.extract()

    assert set(etl._data["terms_to_map"]["term"]) == {"c", "d"}


def test_extract_caps_sample_size(etl: TermEntityMapping, mock_metrics: MagicMock) -> None:
    """extract should cap the sampling to sample_size."""
    df_terms = pd.DataFrame({"term": [f"t{i}" for i in range(100)]})
    etl._store.get.side_effect = [MagicMock(train=df_terms), NoStoreException("missing")]

    etl.extract()

    assert etl._data["terms_to_map"].shape[0] == 10


def test_extract_handles_missing_silver_table(etl: TermEntityMapping, mock_metrics: MagicMock) -> None:
    """extract should map all bronze terms when the silver table does not exist."""
    df_terms = pd.DataFrame({"term": ["a", "b"]})
    etl._store.get.side_effect = [MagicMock(train=df_terms), NoStoreException("missing")]

    etl.extract()

    assert set(etl._data["terms_to_map"]["term"]) == {"a", "b"}


# --------------------------------------------------------------------------- #
# transform
# --------------------------------------------------------------------------- #


def test_transform_maps_terms_to_entities(etl: TermEntityMapping, mock_metrics: MagicMock) -> None:
    """transform should map terms to entities and keep only non-null entities."""
    etl._data["terms_to_map"] = pd.DataFrame({"term": ["aspirin", "unknown"]})

    with patch("tgedr_fdafaers.etl.term_entity_mapping.TermEntity") as mock_term_entity_cls:
        mock_term_entity = MagicMock()
        mock_term_entity_cls.return_value = mock_term_entity
        mock_term_entity.process.return_value = pd.DataFrame(
            {"term": ["aspirin", "unknown"], "entity": ["aspirin", None]}
        )

        etl.transform()

    assert etl._result is not None
    assert list(etl._result["entity"]) == ["aspirin"]


def test_transform_skips_when_no_terms(etl: TermEntityMapping, mock_metrics: MagicMock) -> None:
    """transform should not set a result when there are no terms to map."""
    etl._data["terms_to_map"] = pd.DataFrame({"term": []})

    etl.transform()

    assert etl._result is None


def test_transform_skips_when_all_entities_null(etl: TermEntityMapping, mock_metrics: MagicMock) -> None:
    """transform should not set a result when all mapped entities are null."""
    etl._data["terms_to_map"] = pd.DataFrame({"term": ["a", "b"]})

    with patch("tgedr_fdafaers.etl.term_entity_mapping.TermEntity") as mock_term_entity_cls:
        mock_term_entity = MagicMock()
        mock_term_entity_cls.return_value = mock_term_entity
        mock_term_entity.process.return_value = pd.DataFrame({"term": ["a", "b"], "entity": [None, None]})

        etl.transform()

    assert etl._result is None


# --------------------------------------------------------------------------- #
# load
# --------------------------------------------------------------------------- #


def test_load_saves_result(etl: TermEntityMapping, mock_metrics: MagicMock) -> None:
    """load should save the result and record the new rows metric."""
    etl._result = pd.DataFrame({"term": ["aspirin"], "entity": ["aspirin"]})

    etl.load()

    etl._store.save.assert_called_once_with(
        df=etl._result, key="s/term_entity", split="train", append=True
    )
    mock_metrics.add_to_gauge.assert_called_once()


def test_load_skips_save_without_result(etl: TermEntityMapping, mock_metrics: MagicMock) -> None:
    """load should not save when there is no result."""
    etl._result = None

    etl.load()

    etl._store.save.assert_not_called()


def test_load_skips_save_with_empty_result(etl: TermEntityMapping, mock_metrics: MagicMock) -> None:
    """load should not save when the result is empty."""
    etl._result = pd.DataFrame({"term": [], "entity": []})

    etl.load()

    etl._store.save.assert_not_called()

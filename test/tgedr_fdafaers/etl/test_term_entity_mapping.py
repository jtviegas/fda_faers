"""Unit tests for the TermEntityMapping ETL."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from tgedr_dataops.store.contracted_store import NoStoreException

from tgedr_fdafaers.etl.term_entity_mapping import TermEntityMapping


@pytest.fixture()
def etl() -> TermEntityMapping:
    """Return a TermEntityMapping instance with a mocked store."""
    with patch("tgedr_fdafaers.etl.term_entity_mapping.HuggingFaceDatasetFileArrowBasedStore") as mock_store_cls:
        mock_store = MagicMock()
        mock_store_cls.return_value = mock_store
        etl = TermEntityMapping(
            configuration={
                "drug_data_url": "b/drug",
                "term_entity_data_url": "s/term_entity",
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
    mock_terms = MagicMock()
    mock_terms.train.unique.return_value = ["a", "b", "c"]
    etl._store.get.side_effect = [mock_terms, NoStoreException("missing")]

    etl.extract()

    assert etl._store.get.call_args_list[0].kwargs["key"] == "b/drug"
    assert etl._store.get.call_args_list[1].kwargs["key"] == "s/term_entity"
    assert set(etl._data["terms_to_map"]) == {"a", "b", "c"}
    mock_metrics.add_to_gauge.assert_called()


def test_extract_excludes_terms_already_mapped(etl: TermEntityMapping, mock_metrics: MagicMock) -> None:
    """extract should exclude terms already present in the silver term_entity table."""
    mock_terms = MagicMock()
    mock_terms.train.unique.return_value = ["a", "b", "c", "d"]
    mock_mapped = MagicMock()
    mock_mapped.train.unique.return_value = ["a", "b"]
    etl._store.get.side_effect = [mock_terms, mock_mapped]

    etl.extract()

    assert set(etl._data["terms_to_map"]) == {"c", "d"}


def test_extract_skips_when_all_terms_already_mapped(etl: TermEntityMapping, mock_metrics: MagicMock) -> None:
    """extract should not store a sample when every bronze term is already mapped."""
    mock_terms = MagicMock()
    mock_terms.train.unique.return_value = ["a", "b"]
    mock_mapped = MagicMock()
    mock_mapped.train.unique.return_value = ["a", "b"]
    etl._store.get.side_effect = [mock_terms, mock_mapped]

    etl.extract()

    assert "terms_to_map" not in etl._data


def test_extract_caps_sample_size(etl: TermEntityMapping, mock_metrics: MagicMock) -> None:
    """extract should cap the sampling to sample_size."""
    mock_terms = MagicMock()
    mock_terms.train.unique.return_value = [f"t{i}" for i in range(100)]
    etl._store.get.side_effect = [mock_terms, NoStoreException("missing")]

    etl.extract()

    assert len(etl._data["terms_to_map"]) == 10


def test_extract_handles_missing_silver_table(etl: TermEntityMapping, mock_metrics: MagicMock) -> None:
    """extract should map all bronze terms when the silver table does not exist."""
    mock_terms = MagicMock()
    mock_terms.train.unique.return_value = ["a", "b"]
    etl._store.get.side_effect = [mock_terms, NoStoreException("missing")]

    etl.extract()

    assert set(etl._data["terms_to_map"]) == {"a", "b"}


# --------------------------------------------------------------------------- #
# transform
# --------------------------------------------------------------------------- #


def test_transform_maps_terms_to_entities(etl: TermEntityMapping, mock_metrics: MagicMock) -> None:
    """transform should map terms to entities and keep only non-null entities."""
    etl._data["terms_to_map"] = ["aspirin", "unknown"]

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
    etl._data["terms_to_map"] = []

    etl.transform()

    assert etl._result is None


def test_transform_skips_when_all_entities_null(etl: TermEntityMapping, mock_metrics: MagicMock) -> None:
    """transform should not set a result when all mapped entities are null."""
    etl._data["terms_to_map"] = ["a", "b"]

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
        df=etl._result, key="s/term_entity", split="train", append=True,
        data_contract=etl.contract_path("term_entity")
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

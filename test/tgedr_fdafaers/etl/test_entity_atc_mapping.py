"""Unit tests for the EntityAtcMapping ETL."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from tgedr_dataops.store.contracted_store import NoStoreException

from tgedr_fdafaers.etl.entity_atc_mapping import EntityAtcMapping


@pytest.fixture()
def etl() -> EntityAtcMapping:
    """Return an EntityAtcMapping instance with a mocked store."""
    with patch("tgedr_fdafaers.etl.entity_atc_mapping.ContractedHFDatasetFileBasedStore") as mock_store_cls:
        mock_store = MagicMock()
        mock_store_cls.return_value = mock_store
        etl = EntityAtcMapping(
            configuration={
                "silver_dataset_prefix": "s/",
                "sample_size": 10,
            }
        )
        etl._store = mock_store
        yield etl


@pytest.fixture()
def mock_metrics() -> MagicMock:
    """Return a mocked Metrics.instance()."""
    with patch("tgedr_fdafaers.etl.entity_atc_mapping.Metrics") as mock_metrics_cls:
        mock_metrics = MagicMock()
        mock_metrics_cls.instance.return_value = mock_metrics
        yield mock_metrics


# --------------------------------------------------------------------------- #
# extract
# --------------------------------------------------------------------------- #


def test_extract_reads_entities_to_map(etl: EntityAtcMapping, mock_metrics: MagicMock) -> None:
    """extract should read silver term_entity and store a sample of unmapped entities."""
    df_entity = pd.DataFrame({"entity": ["a", "b", "c"]})
    etl._store.get.side_effect = [MagicMock(train=df_entity), NoStoreException("missing")]

    etl.extract()

    assert etl._store.get.call_args_list[0].kwargs["key"] == "s/term_entity"
    assert etl._store.get.call_args_list[1].kwargs["key"] == "s/entity_atc"
    assert set(etl._data["df_entities_to_map"]["entity"]) == {"a", "b", "c"}
    mock_metrics.add_to_gauge.assert_called()


def test_extract_excludes_already_mapped_entities(etl: EntityAtcMapping, mock_metrics: MagicMock) -> None:
    """extract should exclude entities already present in the entity_atc table."""
    df_entity = pd.DataFrame({"entity": ["a", "b", "c", "d"]})
    df_mapped = pd.DataFrame({"entity": ["a", "b"]})
    etl._store.get.side_effect = [MagicMock(train=df_entity), MagicMock(train=df_mapped)]

    etl.extract()

    assert set(etl._data["df_entities_to_map"]["entity"]) == {"c", "d"}


def test_extract_caps_sample_size(etl: EntityAtcMapping, mock_metrics: MagicMock) -> None:
    """extract should cap the sampling to sample_size."""
    df_entity = pd.DataFrame({"entity": [f"e{i}" for i in range(100)]})
    etl._store.get.side_effect = [MagicMock(train=df_entity), NoStoreException("missing")]

    etl.extract()

    assert etl._data["df_entities_to_map"].shape[0] == 10


def test_extract_handles_missing_entity_atc_table(etl: EntityAtcMapping, mock_metrics: MagicMock) -> None:
    """extract should map all entities when the entity_atc table does not exist."""
    df_entity = pd.DataFrame({"entity": ["a", "b"]})
    etl._store.get.side_effect = [MagicMock(train=df_entity), NoStoreException("missing")]

    etl.extract()

    assert set(etl._data["df_entities_to_map"]["entity"]) == {"a", "b"}


# --------------------------------------------------------------------------- #
# transform
# --------------------------------------------------------------------------- #


def test_transform_maps_entities_to_atc(etl: EntityAtcMapping, mock_metrics: MagicMock) -> None:
    """transform should decorate entities with ATC codes and keep only mapped ones."""
    etl._data["df_entities_to_map"] = pd.DataFrame({"entity": ["aspirin", "zzz"]})

    with patch("tgedr_fdafaers.etl.entity_atc_mapping.NLMMapping") as mock_nlm_cls:
        mock_nlm = MagicMock()
        mock_nlm_cls.return_value = mock_nlm
        mock_nlm.decorate.return_value = pd.DataFrame(
            {
                "entity": ["aspirin", "zzz"],
                "atc_codes": [[{"id": "A01A"}], None],
                "strategy": ["NLMMapping", "NLMMapping"],
            }
        )

        etl.transform()

    mock_nlm_cls.assert_called_once_with(term_col="entity")
    assert etl._result is not None
    assert list(etl._result["entity"]) == ["aspirin"]


def test_transform_skips_when_no_entities(etl: EntityAtcMapping, mock_metrics: MagicMock) -> None:
    """transform should not set a result when there are no entities to map."""
    etl._data = {}

    etl.transform()

    assert etl._result is None


def test_transform_skips_when_all_unmapped(etl: EntityAtcMapping, mock_metrics: MagicMock) -> None:
    """transform should not set a result when no entity got mapped."""
    etl._data["df_entities_to_map"] = pd.DataFrame({"entity": ["zzz"]})

    with patch("tgedr_fdafaers.etl.entity_atc_mapping.NLMMapping") as mock_nlm_cls:
        mock_nlm = MagicMock()
        mock_nlm_cls.return_value = mock_nlm
        mock_nlm.decorate.return_value = pd.DataFrame(
            {"entity": ["zzz"], "atc_codes": [None], "strategy": ["NLMMapping"]}
        )

        etl.transform()

    assert etl._result is None


# --------------------------------------------------------------------------- #
# load
# --------------------------------------------------------------------------- #


def test_load_saves_result(etl: EntityAtcMapping, mock_metrics: MagicMock) -> None:
    """load should save the result and record the new rows metric."""
    etl._result = pd.DataFrame({"entity": ["aspirin"], "atc_codes": [[{"id": "A01A"}]]})

    etl.load()

    etl._store.save.assert_called_once_with(
        df=etl._result, key="s/entity_atc", split="train", append=True
    )
    mock_metrics.add_to_gauge.assert_called_once()


def test_load_skips_save_without_result(etl: EntityAtcMapping, mock_metrics: MagicMock) -> None:
    """load should not save when there is no result."""
    etl._result = None

    etl.load()

    etl._store.save.assert_not_called()


def test_load_skips_save_with_empty_result(etl: EntityAtcMapping, mock_metrics: MagicMock) -> None:
    """load should not save when the result is empty."""
    etl._result = pd.DataFrame({"entity": [], "atc_codes": []})

    etl.load()

    etl._store.save.assert_not_called()

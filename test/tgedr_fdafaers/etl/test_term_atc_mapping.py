"""Unit tests for the TermAtcMapping ETL."""

# ruff: noqa: SLF001

from unittest.mock import MagicMock, patch

import pytest
from datasets import Dataset

from tgedr_dataops.store.hf_dataset_file_arrow_based import NoStoreException
from tgedr_fdafaers.etl.term_atc_mapping import TermAtcMapping


@pytest.fixture
def etl() -> TermAtcMapping:
    """Create a TermAtcMapping instance with a mocked dataset store."""
    with patch("tgedr_fdafaers.etl.term_atc_mapping.HuggingFaceDatasetFileArrowBasedStore") as store_class:
        store = MagicMock()
        store_class.return_value = store
        instance = TermAtcMapping(configuration={"silver_dataset_prefix": "silver/"})
        instance._store = store
        yield instance


@pytest.fixture
def metrics() -> MagicMock:
    """Mock metric recording for the ETL."""
    with patch("tgedr_fdafaers.etl.term_atc_mapping.Metrics") as metrics_class:
        instance = MagicMock()
        metrics_class.instance.return_value = instance
        yield instance


def test_extract_reads_both_mapping_datasets(etl: TermAtcMapping) -> None:
    """Load term-entity as streaming and entity-ATC as a materialized dataset."""
    term_entity = Dataset.from_list([{"term": "headache", "entity": "headache"}])
    entity_atc = Dataset.from_list([{"entity": "headache", "atc_codes": [{"id": "N02"}]}])
    etl._store.get.side_effect = [
        MagicMock(train=Dataset.from_list([{"term": "old", "atc_codes": []}])),
        MagicMock(train=term_entity),
        MagicMock(train=entity_atc),
    ]

    etl.extract()

    assert etl._store.get.call_args_list[0].kwargs == {"key": "silver/term_atc", "streaming": False}
    assert etl._store.get.call_args_list[1].kwargs == {"key": "silver/term_entity", "streaming": False}
    assert etl._store.get.call_args_list[2].kwargs == {"key": "silver/entity_atc", "streaming": False}
    assert list(etl._data["term_entity"]) == list(term_entity)
    assert list(etl._data["entity_atc"]) == list(entity_atc)


def test_extract_treats_missing_existing_mapping_as_no_terms_mapped(
    etl: TermAtcMapping, metrics: MagicMock
) -> None:
    """Load all terms when the existing term-ATC dataset is absent."""
    term_entity = Dataset.from_list(
        [{"term": "headache", "entity": "headache"}, {"term": "pain", "entity": "pain"}]
    )
    entity_atc = Dataset.from_list(
        [{"entity": "headache", "atc_codes": [{"id": "N02"}]}, {"entity": "pain", "atc_codes": []}]
    )
    etl._store.get.side_effect = [
        NoStoreException("not found"),
        MagicMock(train=term_entity),
        MagicMock(train=entity_atc),
    ]

    etl.extract()

    assert list(etl._data["term_entity"]) == list(term_entity)
    assert list(etl._data["entity_atc"]) == list(entity_atc)
    assert metrics.add_to_gauge.call_count == 2
    metrics.add_to_gauge.assert_any_call(
        name="fda_faers.term_atc_mapping.rows", value=0, attributes={"type": "term_atc_mapped"}
    )
    metrics.add_to_gauge.assert_any_call(
        name="fda_faers.term_atc_mapping.rows", value=2, attributes={"type": "term_atc_unmapped"}
    )


def test_extract_excludes_terms_already_mapped(etl: TermAtcMapping, metrics: MagicMock) -> None:
    """Retain only terms not already in term_atc and entities used by those terms."""
    existing = Dataset.from_list([{"term": "already mapped", "atc_codes": [{"id": "N02"}]}])
    term_entity = Dataset.from_list(
        [
            {"term": "already mapped", "entity": "mapped entity"},
            {"term": "new term", "entity": "new entity"},
        ]
    )
    entity_atc = Dataset.from_list(
        [
            {"entity": "mapped entity", "atc_codes": [{"id": "N02"}]},
            {"entity": "new entity", "atc_codes": [{"id": "N03"}]},
            {"entity": "unused entity", "atc_codes": [{"id": "N04"}]},
        ]
    )
    etl._store.get.side_effect = [
        MagicMock(train=existing),
        MagicMock(train=term_entity),
        MagicMock(train=entity_atc),
    ]

    etl.extract()

    assert etl._data["term_entity"]["term"] == ["new term"]
    assert etl._data["entity_atc"]["entity"] == ["new entity"]
    assert metrics.add_to_gauge.call_count == 2


def test_transform_maps_and_accumulates_atc_codes(etl: TermAtcMapping, metrics: MagicMock) -> None:
    """Combine ATC codes for multiple entities linked to the same term."""
    etl._data = {
        "term_entity": Dataset.from_list(
            [
                {"term": "headache", "entity": "headache"},
                {"term": "headache", "entity": "cephalalgia"},
                {"term": "unknown", "entity": "not mapped"},
            ]
        ),
        "entity_atc": Dataset.from_list(
            [
                {"entity": "headache", "atc_codes": [{"id": "N02"}]},
                {"entity": "cephalalgia", "atc_codes": [{"id": "N02B"}]},
            ]
        ),
    }

    etl.transform()

    assert etl._result is not None
    rows = {row["term"]: row["atc_codes"] for row in etl._result}
    assert rows == {"headache": [{"id": "N02"}, {"id": "N02B"}]}
    assert metrics.add_to_gauge.call_count == 1
    metrics.add_to_gauge.assert_called_once_with(
        name="fda_faers.term_atc_mapping.rows",
        value=1,
        attributes={"type": "term_atc_mapped_new"},
    )


def test_transform_leaves_result_unset_when_no_terms(etl: TermAtcMapping, metrics: MagicMock) -> None:
    """Leave the result unset and skip metrics when there are no term rows."""
    etl._data = {
        "term_entity": Dataset.from_list([]),
        "entity_atc": Dataset.from_list([]),
    }

    etl.transform()

    assert etl._result is None
    metrics.add_to_gauge.assert_not_called()


def test_load_saves_nonempty_result(etl: TermAtcMapping) -> None:
    """Persist a populated result under the term_atc dataset key."""
    result = Dataset.from_list([{"term": "headache", "atc_codes": [{"id": "N02"}]}])
    etl._result = result

    etl.load()

    etl._store.save.assert_called_once_with(df=result, key="silver/term_atc", split="train", append=True)


@pytest.mark.parametrize("result", [None, Dataset.from_list([{"term": "headache", "atc_codes": []}]).select([])])
def test_load_skips_missing_or_empty_result(etl: TermAtcMapping, result: Dataset | None) -> None:
    """Do not write a result that is missing or has no rows."""
    etl._result = result

    etl.load()

    etl._store.save.assert_not_called()

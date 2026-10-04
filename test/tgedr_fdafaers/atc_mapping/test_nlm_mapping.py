"""Unit tests for the NLM ATC mapping module."""

from typing import Any

import pandas as pd
import pytest

from tgedr_fdafaers.atc_mapping.mapping import AtcClass
from tgedr_fdafaers.atc_mapping.nlm_mapping import NLMMapping
from tgedr_fdafaers.atc_mapping.nlm_rxnav_api.facade import NLMRxNavApiFacade


# --------------------------------------------------------------------------- #
# map_atc_classes
# --------------------------------------------------------------------------- #


class _FakeFacade:
    """Configurable stand-in for NLMRxNavApiFacade."""

    def __init__(
        self,
        rxnormid: str | None = "1234",
        concepts: list[Any] | None = None,
        history_ingredients: list[Any] | None = None,
        atc_classes: list[Any] | None = None,
    ) -> None:
        self._rxnormid = rxnormid
        self._concepts = concepts if concepts is not None else []
        self._history_ingredients = history_ingredients if history_ingredients is not None else []
        self._atc_classes = atc_classes if atc_classes is not None else []
        self.concepts_calls: list[Any] = []
        self.history_calls: list[Any] = []

    def get_rxnormid(self, term: str) -> str | None:
        return self._rxnormid

    def get_concepts(self, rxnormid: str, tty_filter: list[str] | None = None) -> list[Any]:
        self.concepts_calls.append(rxnormid)
        return self._concepts

    def get_ingredients_from_history(self, rxnormid: str) -> list[Any]:
        self.history_calls.append(rxnormid)
        return self._history_ingredients

    def get_atc_classes(self, rxcui: str) -> list[Any]:
        return self._atc_classes


def _mapping(facade: _FakeFacade) -> NLMMapping:
    """Build an NLMMapping wired to the given fake facade."""
    return NLMMapping(facade=facade)  # type: ignore[arg-type]


def test_map_atc_classes_returns_empty_when_no_rxnormid() -> None:
    """A missing RxNorm ID should yield an empty list."""
    facade = _FakeFacade(rxnormid=None)

    assert _mapping(facade).map_atc_classes("unknown") == []


def test_map_atc_classes_returns_empty_when_no_ingredients() -> None:
    """No ingredients (direct or from history) should yield an empty list."""
    facade = _FakeFacade(concepts=[], history_ingredients=[])

    assert _mapping(facade).map_atc_classes("aspirin") == []


def test_map_atc_classes_falls_back_to_history() -> None:
    """When direct concepts are empty, history ingredients should be used."""
    facade = _FakeFacade(
        concepts=[],
        history_ingredients=[{"rxcui": "100", "name": "Ing", "tty": "IN"}],
        atc_classes=[{"classId": "A01", "className": "X", "classType": "ATC"}],
    )

    result = _mapping(facade).map_atc_classes("aspirin")

    assert facade.history_calls == ["1234"]
    assert len(result) == 1
    assert result[0].id == "A01"
    assert result[0].name == "X"
    assert result[0].class_type == "ATC"


def test_map_atc_classes_uses_direct_ingredients() -> None:
    """Direct ingredients should be used without falling back to history."""
    facade = _FakeFacade(
        concepts=[
            {"rxcui": "100", "name": "IngA", "tty": "IN"},
            {"rxcui": "200", "name": "IngB", "tty": "PIN"},
        ],
        atc_classes=[{"classId": "A01", "className": "X", "classType": "ATC"}],
    )

    result = _mapping(facade).map_atc_classes("aspirin")

    assert facade.history_calls == []
    assert len(result) == 2


def test_map_atc_classes_aggregates_multiple_classes_per_ingredient() -> None:
    """Multiple ATC classes per ingredient should all be collected."""
    facade = _FakeFacade(
        concepts=[{"rxcui": "100", "name": "IngA", "tty": "IN"}],
        atc_classes=[
            {"classId": "A01", "className": "X", "classType": "ATC"},
            {"classId": "A02", "className": "Y", "classType": "ATC"},
        ],
    )

    result = _mapping(facade).map_atc_classes("aspirin")

    assert len(result) == 2
    assert {c.id for c in result} == {"A01", "A02"}


# --------------------------------------------------------------------------- #
# NLMMapping
# --------------------------------------------------------------------------- #


def test_nlm_mapping_init_defaults() -> None:
    """Default column names should be term/atc_codes."""
    mapping = NLMMapping()

    assert mapping._term_col == "term"
    assert mapping._output_col == "atc_codes"


def test_nlm_mapping_init_custom_columns() -> None:
    """Custom column names should be honoured."""
    mapping = NLMMapping(term_col="drug", output_col="codes")

    assert mapping._term_col == "drug"
    assert mapping._output_col == "codes"


def test_nlm_mapping_init_creates_default_facade() -> None:
    """Without a facade argument a real NLMRxNavApiFacade should be created."""
    mapping = NLMMapping()

    assert isinstance(mapping._facade, NLMRxNavApiFacade)


def test_nlm_mapping_init_accepts_injected_facade() -> None:
    """An injected facade should be reused instead of creating a new one."""
    facade = _FakeFacade()
    mapping = NLMMapping(facade=facade)  # type: ignore[arg-type]

    assert mapping._facade is facade


def test_decorate_adds_columns_and_preserves_rows() -> None:
    """decorate should add output and strategy columns without dropping rows."""
    df = pd.DataFrame({"term": ["aspirin", "aspirin", "ibuprofen", ""]})
    mapping = _mapping(
        _FakeFacade(
            concepts=[{"rxcui": "100", "name": "IngA", "tty": "IN"}],
            atc_classes=[{"classId": "A01", "className": "X", "classType": "ATC"}],
        )
    )

    result = mapping.decorate(df)

    assert "atc_codes" in result.columns
    assert "strategy" in result.columns
    assert len(result) == 4
    assert result["strategy"].iloc[0] == "NLMMapping"
    assert result["atc_codes"].iloc[0] == [{"id": "A01", "name": "X", "class_type": "ATC"}]
    assert result["atc_codes"].iloc[3] is None


def test_decorate_returns_none_for_empty_term() -> None:
    """An empty term should resolve to None in the output column."""
    df = pd.DataFrame({"term": [""]})
    mapping = _mapping(
        _FakeFacade(
            concepts=[{"rxcui": "100", "name": "IngA", "tty": "IN"}],
            atc_classes=[{"classId": "A01", "className": "X", "classType": "ATC"}],
        )
    )

    result = mapping.decorate(df)

    assert result["atc_codes"].iloc[0] is None


def test_decorate_returns_none_when_no_classes() -> None:
    """A term with no resolved classes should yield None in the output column."""
    df = pd.DataFrame({"term": ["aspirin"]})
    mapping = _mapping(_FakeFacade(atc_classes=[]))

    result = mapping.decorate(df)

    assert result["atc_codes"].iloc[0] is None


def test_decorate_deduplicates_api_calls() -> None:
    """Duplicate terms should only be resolved once."""
    df = pd.DataFrame({"term": ["aspirin", "aspirin", "aspirin"]})
    facade = _FakeFacade(
        concepts=[{"rxcui": "100", "name": "IngA", "tty": "IN"}],
        atc_classes=[{"classId": "A01", "className": "X", "classType": "ATC"}],
    )

    result = _mapping(facade).decorate(df)

    assert len(facade.concepts_calls) == 1
    assert len(result) == 3

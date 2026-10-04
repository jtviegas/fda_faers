"""Unit tests for the NLM RxNav API facade."""

from typing import Any

import pytest
import requests

from tgedr_fdafaers.atc_mapping.nlm_rxnav_api.facade import (
    AtcMappingError,
    NLMRxNavApiFacade,
)


class _FakeResponse:
    """Minimal stand-in for requests.Response."""

    def __init__(self, status_code: int, payload: Any = None) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> Any:
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


@pytest.fixture()
def facade() -> NLMRxNavApiFacade:
    """Provide a fresh facade instance for each test."""
    return NLMRxNavApiFacade()


def _patch_get(monkeypatch: pytest.MonkeyPatch, response: _FakeResponse) -> None:
    """Patch requests.get to always return the given fake response."""

    def fake_get(*args: Any, **kwargs: Any) -> _FakeResponse:
        return response

    monkeypatch.setattr(
        "tgedr_fdafaers.atc_mapping.nlm_rxnav_api.facade.requests.get",
        fake_get,
    )


# --------------------------------------------------------------------------- #
# _get_json
# --------------------------------------------------------------------------- #


def test_get_json_returns_payload(facade: NLMRxNavApiFacade, monkeypatch: pytest.MonkeyPatch) -> None:
    """A successful 200 response should return the parsed JSON payload."""
    _patch_get(monkeypatch, _FakeResponse(200, {"key": "value"}))

    assert facade._get_json("https://example.com") == {"key": "value"}


def test_get_json_raises_on_request_exception(facade: NLMRxNavApiFacade, monkeypatch: pytest.MonkeyPatch) -> None:
    """Network failures should be wrapped in AtcMappingError."""

    def fake_get(*args: Any, **kwargs: Any) -> None:
        raise requests.ConnectionError("boom")

    monkeypatch.setattr(
        "tgedr_fdafaers.atc_mapping.nlm_rxnav_api.facade.requests.get",
        fake_get,
    )

    with pytest.raises(AtcMappingError, match="request failed"):
        facade._get_json("https://example.com")


def test_get_json_raises_on_server_error(facade: NLMRxNavApiFacade, monkeypatch: pytest.MonkeyPatch) -> None:
    """A 5xx response should raise AtcMappingError."""
    _patch_get(monkeypatch, _FakeResponse(500, None))

    with pytest.raises(AtcMappingError, match="server error"):
        facade._get_json("https://example.com")


def test_get_json_returns_none_on_client_error(facade: NLMRxNavApiFacade, monkeypatch: pytest.MonkeyPatch) -> None:
    """A 4xx response should return None so callers can fall back to defaults."""
    _patch_get(monkeypatch, _FakeResponse(404, None))

    assert facade._get_json("https://example.com") is None


def test_get_json_raises_on_invalid_json(facade: NLMRxNavApiFacade, monkeypatch: pytest.MonkeyPatch) -> None:
    """An unparseable payload should raise AtcMappingError."""
    _patch_get(monkeypatch, _FakeResponse(200, ValueError("bad json")))

    with pytest.raises(AtcMappingError, match="invalid JSON"):
        facade._get_json("https://example.com")


# --------------------------------------------------------------------------- #
# get_rxnormid
# --------------------------------------------------------------------------- #


def test_get_rxnormid_returns_candidate(facade: NLMRxNavApiFacade, monkeypatch: pytest.MonkeyPatch) -> None:
    """The first candidate rxcui should be returned."""
    _patch_get(
        monkeypatch,
        _FakeResponse(200, {"approximateGroup": {"candidate": [{"rxcui": "1234"}]}}),
    )

    assert facade.get_rxnormid("aspirin") == "1234"


def test_get_rxnormid_returns_none_without_candidates(
    facade: NLMRxNavApiFacade, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty candidate list should yield None."""
    _patch_get(monkeypatch, _FakeResponse(200, {"approximateGroup": {"candidate": []}}))

    assert facade.get_rxnormid("aspirin") is None


def test_get_rxnormid_returns_none_on_client_error(
    facade: NLMRxNavApiFacade, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 4xx response should yield None."""
    _patch_get(monkeypatch, _FakeResponse(404, None))

    assert facade.get_rxnormid("aspirin") is None


# --------------------------------------------------------------------------- #
# get_concepts
# --------------------------------------------------------------------------- #


def test_get_concepts_returns_all_without_filter(
    facade: NLMRxNavApiFacade, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without a tty filter all concept groups should be returned."""
    payload = {
        "allRelatedGroup": {
            "conceptGroup": [
                {"tty": "IN", "conceptProperties": [{"rxcui": "1", "name": "A", "tty": "IN"}]},
                {"tty": "SCD", "conceptProperties": [{"rxcui": "2", "name": "B", "tty": "SCD"}]},
            ]
        }
    }
    _patch_get(monkeypatch, _FakeResponse(200, payload))

    result = facade.get_concepts("1234")

    assert len(result) == 2


def test_get_concepts_filters_by_tty(facade: NLMRxNavApiFacade, monkeypatch: pytest.MonkeyPatch) -> None:
    """Only concept groups matching the tty filter should be returned."""
    payload = {
        "allRelatedGroup": {
            "conceptGroup": [
                {"tty": "IN", "conceptProperties": [{"rxcui": "1", "name": "A", "tty": "IN"}]},
                {"tty": "SCD", "conceptProperties": [{"rxcui": "2", "name": "B", "tty": "SCD"}]},
            ]
        }
    }
    _patch_get(monkeypatch, _FakeResponse(200, payload))

    result = facade.get_concepts("1234", tty_filter=["IN"])

    assert len(result) == 1
    assert result[0]["rxcui"] == "1"


def test_get_concepts_deduplicates_by_rxcui(facade: NLMRxNavApiFacade, monkeypatch: pytest.MonkeyPatch) -> None:
    """Duplicate rxcui entries should be collapsed into a single concept."""
    payload = {
        "allRelatedGroup": {
            "conceptGroup": [
                {"tty": "IN", "conceptProperties": [{"rxcui": "1", "name": "A", "tty": "IN"}]},
                {"tty": "IN", "conceptProperties": [{"rxcui": "1", "name": "A", "tty": "IN"}]},
            ]
        }
    }
    _patch_get(monkeypatch, _FakeResponse(200, payload))

    result = facade.get_concepts("1234")

    assert len(result) == 1


def test_get_concepts_returns_empty_on_client_error(
    facade: NLMRxNavApiFacade, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 4xx response should yield an empty list."""
    _patch_get(monkeypatch, _FakeResponse(404, None))

    assert facade.get_concepts("1234") == []


# --------------------------------------------------------------------------- #
# get_history
# --------------------------------------------------------------------------- #


def test_get_history_parses_full_payload(facade: NLMRxNavApiFacade, monkeypatch: pytest.MonkeyPatch) -> None:
    """All derived concept types should be parsed into concepts and ingredients."""
    payload = {
        "rxcuiStatusHistory": {
            "metaData": {"status": "ACTIVE"},
            "definitionalFeatures": {
                "ingredientAndStrength": [
                    {"activeIngredientRxcui": "10", "activeIngredientName": "IngA"}
                ]
            },
            "derivedConcepts": {
                "ingredientConcept": [{"ingredientRxcui": "11", "ingredientName": "IngB"}],
                "quantifiedConcept": [
                    {
                        "quantifiedRxcui": "20",
                        "quantifiedName": "Q",
                        "quantifiedTTY": "SCD",
                        "quantifiedActive": "true",
                    }
                ],
                "remappedConcept": [
                    {
                        "remappedRxCui": "21",
                        "remappedName": "R",
                        "remappedTTY": "SCD",
                        "remappedActive": "true",
                    }
                ],
                "scdConcept": {"scdConceptRxcui": "22", "scdConceptName": "S"},
                "qdFreeConcept": {"qdFreeRxcui": "23", "qdFreeName": "QD"},
            },
        }
    }
    _patch_get(monkeypatch, _FakeResponse(200, payload))

    result = facade.get_history("1234")

    assert result["status"] == "active"
    assert set(result["ingredients"]) == {"10", "11"}
    assert set(result["concepts"]) == {"20", "21", "22", "23"}
    assert result["concepts"]["22"]["tty"] == "SCD"
    assert result["concepts"]["23"]["tty"] == "QD"


def test_get_history_returns_defaults_when_status_history_is_null(
    facade: NLMRxNavApiFacade, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A null rxcuiStatusHistory should yield empty defaults instead of crashing."""
    _patch_get(monkeypatch, _FakeResponse(200, {"rxcuiStatusHistory": None}))

    assert facade.get_history("1234") == {"status": None, "concepts": {}, "ingredients": {}}


def test_get_history_returns_defaults_on_client_error(
    facade: NLMRxNavApiFacade, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 4xx response should yield empty defaults."""
    _patch_get(monkeypatch, _FakeResponse(404, None))

    assert facade.get_history("1234") == {"status": None, "concepts": {}, "ingredients": {}}


def test_get_history_skips_derived_concepts_when_missing(
    facade: NLMRxNavApiFacade, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A null derivedConcepts block should be skipped gracefully."""
    payload = {
        "rxcuiStatusHistory": {
            "metaData": {"status": "ACTIVE"},
            "definitionalFeatures": {"ingredientAndStrength": []},
            "derivedConcepts": None,
        }
    }
    _patch_get(monkeypatch, _FakeResponse(200, payload))

    result = facade.get_history("1234")

    assert result["status"] == "active"
    assert result["concepts"] == {}
    assert result["ingredients"] == {}


def test_get_history_skips_missing_scd_and_qd_concepts(
    facade: NLMRxNavApiFacade, monkeypatch: pytest.MonkeyPatch
) -> None:
    """derivedConcepts without scdConcept/qdFreeConcept should not add concepts."""
    payload = {
        "rxcuiStatusHistory": {
            "metaData": {"status": "ACTIVE"},
            "definitionalFeatures": {"ingredientAndStrength": []},
            "derivedConcepts": {
                "ingredientConcept": [],
                "quantifiedConcept": [],
                "remappedConcept": [],
            },
        }
    }
    _patch_get(monkeypatch, _FakeResponse(200, payload))

    result = facade.get_history("1234")

    assert result["concepts"] == {}


# --------------------------------------------------------------------------- #
# get_atc_classes
# --------------------------------------------------------------------------- #


def test_get_atc_classes_returns_classes(facade: NLMRxNavApiFacade, monkeypatch: pytest.MonkeyPatch) -> None:
    """ATC classes should be collected, skipping entries without a concept item."""
    payload = {
        "rxclassDrugInfoList": {
            "rxclassDrugInfo": [
                {"rxclassMinConceptItem": {"classId": "A01", "className": "X", "classType": "ATC"}},
                {"rxclassMinConceptItem": None},
            ]
        }
    }
    _patch_get(monkeypatch, _FakeResponse(200, payload))

    result = facade.get_atc_classes("1234")

    assert len(result) == 1
    assert result[0]["classId"] == "A01"


def test_get_atc_classes_returns_empty_on_client_error(
    facade: NLMRxNavApiFacade, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 4xx response should yield an empty list."""
    _patch_get(monkeypatch, _FakeResponse(404, None))

    assert facade.get_atc_classes("1234") == []


# --------------------------------------------------------------------------- #
# find_concept_ingredients
# --------------------------------------------------------------------------- #


def test_find_concept_ingredients_expands_scd_and_qd(
    facade: NLMRxNavApiFacade, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCD/QD concepts should be expanded into their ingredient concepts."""
    payload = {
        "allRelatedGroup": {
            "conceptGroup": [
                {"tty": "IN", "conceptProperties": [{"rxcui": "100", "name": "Ing", "tty": "IN"}]},
            ]
        }
    }
    _patch_get(monkeypatch, _FakeResponse(200, payload))
    concepts = {
        "1": {"rxcui": "1", "name": "SCD1", "tty": "SCD"},
        "2": {"rxcui": "2", "name": "QD1", "tty": "QD"},
    }

    result = facade.find_concept_ingredients(concepts)

    assert len(result) == 1
    assert result[0]["rxcui"] == "100"


def test_find_concept_ingredients_prefers_direct_ingredients(
    facade: NLMRxNavApiFacade, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Direct IN/PIN concepts should be collected without extra API calls."""
    concepts = {
        "1": {"rxcui": "1", "name": "IngA", "tty": "IN"},
        "2": {"rxcui": "2", "name": "IngB", "tty": "PIN"},
    }

    result = facade.find_concept_ingredients(concepts)

    assert len(result) == 2


def test_find_concept_ingredients_direct_overrides_derived(
    facade: NLMRxNavApiFacade, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A direct ingredient should override the same rxcui derived from an SCD."""
    payload = {
        "allRelatedGroup": {
            "conceptGroup": [
                {"tty": "IN", "conceptProperties": [{"rxcui": "100", "name": "Derived", "tty": "IN"}]},
            ]
        }
    }
    _patch_get(monkeypatch, _FakeResponse(200, payload))
    concepts = {
        "1": {"rxcui": "1", "name": "SCD1", "tty": "SCD"},
        "100": {"rxcui": "100", "name": "Direct", "tty": "IN"},
    }

    result = facade.find_concept_ingredients(concepts)

    assert len(result) == 1
    assert result[0]["name"] == "Direct"


def test_find_concept_ingredients_ignores_other_tty(
    facade: NLMRxNavApiFacade, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Concepts with unrelated TTY codes should be ignored."""
    concepts = {"1": {"rxcui": "1", "name": "X", "tty": "BN"}}

    result = facade.find_concept_ingredients(concepts)

    assert result == []


# --------------------------------------------------------------------------- #
# get_ingredients_from_history
# --------------------------------------------------------------------------- #


def test_get_ingredients_from_history_combines_and_deduplicates(
    facade: NLMRxNavApiFacade, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Concept-derived and history ingredients should be combined without duplicates."""
    history_payload = {
        "rxcuiStatusHistory": {
            "metaData": {"status": "ACTIVE"},
            "definitionalFeatures": {
                "ingredientAndStrength": [
                    {"activeIngredientRxcui": "100", "activeIngredientName": "IngA"},
                    {"activeIngredientRxcui": "200", "activeIngredientName": "IngB"},
                ]
            },
            "derivedConcepts": {
                "quantifiedConcept": [
                    {
                        "quantifiedRxcui": "1",
                        "quantifiedName": "SCD1",
                        "quantifiedTTY": "SCD",
                        "quantifiedActive": "true",
                    }
                ],
            },
        }
    }
    concepts_payload = {
        "allRelatedGroup": {
            "conceptGroup": [
                {"tty": "IN", "conceptProperties": [{"rxcui": "100", "name": "IngA", "tty": "IN"}]},
            ]
        }
    }

    def fake_get(url: str, *args: Any, **kwargs: Any) -> _FakeResponse:
        if "historystatus" in url:
            return _FakeResponse(200, history_payload)
        return _FakeResponse(200, concepts_payload)

    monkeypatch.setattr(
        "tgedr_fdafaers.atc_mapping.nlm_rxnav_api.facade.requests.get",
        fake_get,
    )

    result = facade.get_ingredients_from_history("1234")

    assert len(result) == 2
    assert {ingredient["rxcui"] for ingredient in result} == {"100", "200"}

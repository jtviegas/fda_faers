"""Unit tests for the EntityParsing processor and helper functions."""

from typing import Any

import pytest

from tgedr_dataops_abs.processor import Processor, ProcessorException

from tgedr_fdafaers.nlp import entity_parsing
from tgedr_fdafaers.nlp.entity_parsing import (
    EntityParsing,
    parse_entities,
    parse_entities_to_comma_concatenated,
)
from tgedr_fdafaers.nlp.model import Model


class _FakeEntity:
    """Minimal stand-in for a spaCy Span."""

    def __init__(self, text: str) -> None:
        self.text = text


class _FakeDoc:
    """Minimal stand-in for a spaCy Doc."""

    def __init__(self, entities: list[_FakeEntity]) -> None:
        self.ents = entities


class _FakeModel(Model):
    """Minimal Model implementation returning a fixed doc."""

    def __init__(self, doc: _FakeDoc) -> None:
        self._doc = doc

    @property
    def model(self) -> Any:
        return lambda text: self._doc


def _patch_model(monkeypatch: pytest.MonkeyPatch, doc: _FakeDoc) -> None:
    """Patch ModelSingleton so EntityParsing uses the given fake doc."""

    class _FakeSingleton:
        def __init__(self) -> None:
            self.instance = _FakeModel(doc)

    monkeypatch.setattr(entity_parsing, "ModelSingleton", _FakeSingleton)


# --------------------------------------------------------------------------- #
# EntityParsing
# --------------------------------------------------------------------------- #


def test_entity_parsing_is_processor() -> None:
    """EntityParsing should implement the Processor interface."""
    assert issubclass(EntityParsing, Processor)


def test_process_returns_entity_texts(monkeypatch: pytest.MonkeyPatch) -> None:
    """process should return the text of each detected entity."""
    doc = _FakeDoc([_FakeEntity("aspirin"), _FakeEntity("ibuprofen")])
    _patch_model(monkeypatch, doc)

    result = EntityParsing().process({EntityParsing.CONTEXT_KEY_TEXT: "some text"})

    assert result == ["aspirin", "ibuprofen"]


def test_process_returns_empty_list_without_entities(monkeypatch: pytest.MonkeyPatch) -> None:
    """process should return an empty list when no entities are found."""
    _patch_model(monkeypatch, _FakeDoc([]))

    result = EntityParsing().process({EntityParsing.CONTEXT_KEY_TEXT: "some text"})

    assert result == []


def test_process_raises_without_context() -> None:
    """process should raise ProcessorException when context is None."""
    with pytest.raises(ProcessorException, match="text must be provided"):
        EntityParsing().process(None)


def test_process_raises_without_text_key() -> None:
    """process should raise ProcessorException when the text key is missing."""
    with pytest.raises(ProcessorException, match="text must be provided"):
        EntityParsing().process({"other": "value"})


def test_process_raises_with_empty_context() -> None:
    """process should raise ProcessorException for an empty context dict."""
    with pytest.raises(ProcessorException, match="text must be provided"):
        EntityParsing().process({})


# --------------------------------------------------------------------------- #
# parse_entities
# --------------------------------------------------------------------------- #


def test_parse_entities_returns_unique_entities(monkeypatch: pytest.MonkeyPatch) -> None:
    """parse_entities should return unique entity texts."""
    doc = _FakeDoc([_FakeEntity("aspirin"), _FakeEntity("aspirin"), _FakeEntity("ibuprofen")])
    _patch_model(monkeypatch, doc)

    result = parse_entities("some text")

    assert sorted(result) == ["aspirin", "ibuprofen"]


def test_parse_entities_returns_none_for_none() -> None:
    """parse_entities should return None when text is None."""
    assert parse_entities(None) is None


def test_parse_entities_returns_none_for_blank() -> None:
    """parse_entities should return None for blank or whitespace-only text."""
    assert parse_entities("") is None
    assert parse_entities("   ") is None


def test_parse_entities_returns_text_when_no_entities(monkeypatch: pytest.MonkeyPatch) -> None:
    """parse_entities should fall back to the stripped text when no entities are found."""
    _patch_model(monkeypatch, _FakeDoc([]))

    result = parse_entities("  some text  ")

    assert result == ["some text"]


# --------------------------------------------------------------------------- #
# parse_entities_to_comma_concatenated
# --------------------------------------------------------------------------- #


def test_parse_entities_to_comma_concatenated_joins_entities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Entities should be joined into a comma-separated string."""
    doc = _FakeDoc([_FakeEntity("aspirin"), _FakeEntity("ibuprofen")])
    _patch_model(monkeypatch, doc)

    result = parse_entities_to_comma_concatenated("some text")

    # parse_entities returns a set, so ordering is not guaranteed
    assert sorted(result.split(",")) == ["aspirin", "ibuprofen"]


def test_parse_entities_to_comma_concatenated_returns_none_for_none() -> None:
    """Should return None when text is None."""
    assert parse_entities_to_comma_concatenated(None) is None


def test_parse_entities_to_comma_concatenated_returns_none_for_blank() -> None:
    """Should return None for blank or whitespace-only text."""
    assert parse_entities_to_comma_concatenated("") is None
    assert parse_entities_to_comma_concatenated("   ") is None


def test_parse_entities_to_comma_concatenated_returns_text_when_no_entities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Should fall back to the stripped text when no entities are found."""
    _patch_model(monkeypatch, _FakeDoc([]))

    result = parse_entities_to_comma_concatenated("  some text  ")

    assert result == "some text"


def test_parse_entities_to_comma_concatenated_normalizes_whitespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Internal whitespace should be collapsed to single spaces."""
    doc = _FakeDoc([_FakeEntity("aspirin   tablet")])
    _patch_model(monkeypatch, doc)

    result = parse_entities_to_comma_concatenated("some text")

    assert result == "aspirin tablet"


def test_parse_entities_to_comma_concatenated_removes_commas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Commas inside entity text should be removed before joining."""
    doc = _FakeDoc([_FakeEntity("aspirin, tablet")])
    _patch_model(monkeypatch, doc)

    result = parse_entities_to_comma_concatenated("some text")

    assert result == "aspirin tablet"

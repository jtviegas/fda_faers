"""Unit tests for the ModelSingleton."""

from typing import Any

import pytest

from tgedr_pycommons.utils.singleton import SingletonMeta

from tgedr_fdafaers.nlp import model_singleton, spacy_model
from tgedr_fdafaers.nlp.model import Model
from tgedr_fdafaers.nlp.model_singleton import ModelSingleton
from tgedr_fdafaers.nlp.spacy_model import NlpModel


class _FakeModel(Model):
    """Minimal Model implementation for testing."""

    def __init__(self, marker: str = "fake") -> None:
        self.marker = marker

    @property
    def model(self) -> Any:
        return self.marker


@pytest.fixture(autouse=True)
def _reset_singleton() -> None:
    """Clear the cached ModelSingleton before and after each test."""
    SingletonMeta._instances.pop(ModelSingleton, None)
    yield
    SingletonMeta._instances.pop(ModelSingleton, None)


def _patch_nlp_model(monkeypatch: pytest.MonkeyPatch, fake: _FakeModel) -> None:
    """Patch NlpModel so ModelSingleton builds the given fake model."""

    def fake_factory() -> _FakeModel:
        return fake

    monkeypatch.setattr(model_singleton, "NlpModel", fake_factory)


def test_singleton_uses_singleton_meta() -> None:
    """ModelSingleton should use the SingletonMeta metaclass."""
    assert isinstance(ModelSingleton, type)
    assert type(ModelSingleton).__name__ == "SingletonMeta"


def test_init_creates_nlp_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """ModelSingleton.__init__ should create an NlpModel instance."""
    fake = _FakeModel()
    _patch_nlp_model(monkeypatch, fake)

    singleton = ModelSingleton()

    assert isinstance(singleton.instance, _FakeModel)
    assert singleton.instance is fake


def test_instance_returns_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """The instance property should return the cached model."""
    fake = _FakeModel()
    _patch_nlp_model(monkeypatch, fake)

    singleton = ModelSingleton()

    assert singleton.instance is fake


def test_instance_is_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    """Multiple instantiations should return the same instance."""
    fake = _FakeModel()
    _patch_nlp_model(monkeypatch, fake)

    first = ModelSingleton()
    second = ModelSingleton()

    assert first is second
    assert first.instance is second.instance


def test_singleton_uses_real_nlp_model_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without patching NlpModel, ModelSingleton should build a real NlpModel."""
    fake = _FakeModel()
    monkeypatch.setattr(spacy_model.spacy, "load", lambda name: fake)

    singleton = ModelSingleton()

    assert isinstance(singleton.instance, NlpModel)
    assert singleton.instance.model is fake

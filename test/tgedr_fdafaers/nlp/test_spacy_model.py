"""Unit tests for the NlpModel spaCy model implementation."""

import subprocess
import sys
from types import MappingProxyType
from typing import Any

import pytest

from tgedr_fdafaers.nlp import spacy_model
from tgedr_fdafaers.nlp.model import Model
from tgedr_fdafaers.nlp.spacy_model import NlpModel, _SCISPACY_MODELS


class _FakeNlp:
    """Minimal stand-in for a loaded spaCy language model."""

    def __init__(self, name: str = "fake") -> None:
        self.name = name

    def __call__(self, text: str) -> "_FakeDoc":
        return _FakeDoc(text)


class _FakeDoc:
    """Minimal stand-in for a spaCy Doc."""

    def __init__(self, text: str) -> None:
        self.text = text


# --------------------------------------------------------------------------- #
# _SCISPACY_MODELS
# --------------------------------------------------------------------------- #


def test_scispacy_models_is_mapping_proxy() -> None:
    """_SCISPACY_MODELS should be an immutable mapping."""
    assert isinstance(_SCISPACY_MODELS, MappingProxyType)


def test_scispacy_models_contains_en_core_sci_md() -> None:
    """_SCISPACY_MODELS should map en_core_sci_md to a remote URL."""
    assert "en_core_sci_md" in _SCISPACY_MODELS
    assert _SCISPACY_MODELS["en_core_sci_md"].startswith("https://")


def test_scispacy_models_is_immutable() -> None:
    """_SCISPACY_MODELS should not allow mutation."""
    with pytest.raises(TypeError):
        _SCISPACY_MODELS["other"] = "https://example.com"  # type: ignore[index]


# --------------------------------------------------------------------------- #
# _load_model
# --------------------------------------------------------------------------- #


def test_load_model_returns_loaded_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """_load_model should return the model loaded by spacy.load."""
    fake = _FakeNlp()
    monkeypatch.setattr(spacy_model.spacy, "load", lambda name: fake)

    assert NlpModel._load_model("en_core_sci_md") is fake


def test_load_model_installs_and_reloads_on_oserror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_load_model should install the model and retry when spacy.load raises OSError."""
    calls: list[str] = []
    fake = _FakeNlp()

    def fake_load(name: str) -> Any:
        calls.append(name)
        if len(calls) == 1:
            raise OSError("model not found")
        return fake

    installed: list[list[str]] = []

    def fake_check_call(args: list[str]) -> None:
        installed.append(args)

    monkeypatch.setattr(spacy_model.spacy, "load", fake_load)
    monkeypatch.setattr(spacy_model.subprocess, "check_call", fake_check_call)

    result = NlpModel._load_model("en_core_sci_md")

    assert result is fake
    assert calls == ["en_core_sci_md", "en_core_sci_md"]
    assert len(installed) == 1
    assert installed[0][:4] == [sys.executable, "-m", "pip", "install"]
    assert installed[0][4] == _SCISPACY_MODELS["en_core_sci_md"]


def test_load_model_raises_on_unknown_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """_load_model should propagate KeyError for an unknown model name."""
    monkeypatch.setattr(spacy_model.spacy, "load", lambda name: (_ for _ in ()).throw(OSError("missing")))

    with pytest.raises(KeyError):
        NlpModel._load_model("unknown_model")


def test_load_model_propagates_check_call_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_load_model should propagate subprocess failures during install."""

    def fake_load(name: str) -> Any:
        raise OSError("model not found")

    def fake_check_call(args: list[str]) -> None:
        raise subprocess.CalledProcessError(1, args)

    monkeypatch.setattr(spacy_model.spacy, "load", fake_load)
    monkeypatch.setattr(spacy_model.subprocess, "check_call", fake_check_call)

    with pytest.raises(subprocess.CalledProcessError):
        NlpModel._load_model("en_core_sci_md")


# --------------------------------------------------------------------------- #
# NlpModel
# --------------------------------------------------------------------------- #


def test_nlp_model_is_model_subclass() -> None:
    """NlpModel should implement the Model interface."""
    assert issubclass(NlpModel, Model)


def test_init_loads_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """NlpModel.__init__ should load the en_core_sci_md model."""
    fake = _FakeNlp()
    monkeypatch.setattr(spacy_model.spacy, "load", lambda name: fake)

    instance = NlpModel()

    assert instance.model is fake


def test_model_property_returns_loaded_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """The model property should return the loaded spaCy model."""
    fake = _FakeNlp()
    monkeypatch.setattr(spacy_model.spacy, "load", lambda name: fake)

    instance = NlpModel()

    assert instance.model is fake
    assert instance.model.name == "fake"


def test_model_is_callable(monkeypatch: pytest.MonkeyPatch) -> None:
    """The loaded model should be callable on text."""
    fake = _FakeNlp()
    monkeypatch.setattr(spacy_model.spacy, "load", lambda name: fake)

    instance = NlpModel()

    assert instance.model("some text").text == "some text"

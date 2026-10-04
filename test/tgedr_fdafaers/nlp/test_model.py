"""Unit tests for the Model abstract base class."""

import pytest

from tgedr_fdafaers.nlp.model import Model


def test_model_is_abstract() -> None:
    """Model should be an abstract base class."""
    assert issubclass(Model, object)
    assert getattr(Model, "__abstractmethods__", frozenset())


def test_model_cannot_be_instantiated() -> None:
    """Instantiating Model directly should raise TypeError."""
    with pytest.raises(TypeError, match="abstract"):
        Model()


def test_model_property_is_abstract() -> None:
    """The model property should be abstract."""
    assert isinstance(Model.model, property)
    assert Model.model.__isabstractmethod__ is True


def test_model_property_is_read_only() -> None:
    """The model property should not define a setter."""
    assert Model.model.fset is None


def test_subclass_must_implement_model() -> None:
    """A subclass without a model property should remain abstract."""

    class _Incomplete(Model):
        pass

    with pytest.raises(TypeError, match="abstract"):
        _Incomplete()


def test_concrete_subclass_is_instantiable() -> None:
    """A subclass implementing the model property should be instantiable."""

    class _Concrete(Model):
        @property
        def model(self) -> object:
            return object()

    instance = _Concrete()

    assert isinstance(instance, Model)
    assert isinstance(instance.model, object)

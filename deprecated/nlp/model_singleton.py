"""Module for singleton-based Model instance management."""

from tgedr_pycommons.utils.singleton import SingletonMeta
from pvprototypes_faers.nlp.model import Model
from pvprototypes_faers.nlp.spacy_model import NlpModel


class ModelSingleton(metaclass=SingletonMeta):
    """Singleton factory for creating and caching a Model instance."""

    def __init__(self):
        """Initialize the ModelSingleton with an NlpModel instance."""
        self.__model: Model = NlpModel()

    @property
    def instance(self) -> Model:
        """Get the Model instance.

        Returns:
            Model: The cached NlpModel instance.
        """
        return self.__model

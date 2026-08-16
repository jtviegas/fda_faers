# noqa: D100
from abc import ABC, abstractmethod


class Model(ABC):
    """Abstract base class for NLP models."""

    @property
    @abstractmethod
    def model(self):
        """Get the underlying NLP model instance."""

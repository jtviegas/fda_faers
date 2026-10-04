# noqa: D100
from abc import ABC, abstractmethod
from typing import Any


class Model(ABC):
    """Abstract base class for NLP models."""

    @property
    @abstractmethod
    def model(self) -> Any:
        """Get the underlying NLP model instance."""

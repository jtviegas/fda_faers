"""File store abstraction."""
from abc import ABC, abstractmethod
from typing import Any


class FileStoreError(Exception):
    """Exception raised for store-related errors."""

class FileStore(ABC):
    """Abstract interface for storing and managing files."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initializes the FileStore instance.

        Args:
            config: Optional configuration dictionary.
        """
        self.config = config

    @abstractmethod
    def list(self, key: str, **kwargs) -> list[str]:
        """Lists files at the given key."""
        raise NotImplementedError

    @abstractmethod
    def get(self, key: str, **kwargs) -> None:
        """Downloads a file from the store to the local filesystem."""
        raise NotImplementedError

    @abstractmethod
    def delete(self, key: str, **kwargs) -> None:
        """Deletes a file from the store."""
        raise NotImplementedError

    @abstractmethod
    def put(self, source: str, target: str) -> None:
        """Uploads a file from the local filesystem to the store."""
        raise NotImplementedError

"""I/O utility helpers."""

import logging
from typing import ClassVar
import urllib.request
from urllib.error import HTTPError
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

class UtilsIOError(Exception):
    """Exception raised for UtilsIO-related errors."""

class UtilsIO:
    """Utility methods for I/O-related operations."""

    __ALLOWED_SCHEMES: ClassVar[set[str]] = {"http", "https", "dbfs"}

    @staticmethod
    def resource_exists(url: str) -> bool:
        """Return the resolved URL if the file exists (HTTP HEAD), else None."""
        logger.info(f"[resource_exists|in] ({url})")
        result: bool = False

        scheme = urlparse(url).scheme
        if 0 < len(scheme) and scheme not in UtilsIO.__ALLOWED_SCHEMES:
            raise UtilsIOError(f"[resource_exists] unsupported URL scheme: {scheme} - {url}")

        try:
            req = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(req) as response:   # nosec B310
                result = (200 == response.code)
        except HTTPError as x:
            raise UtilsIOError(f"[resource_exists] failed request to: {url}") from x

        logger.info(f"[resource_exists|out] => {result}")

        return result

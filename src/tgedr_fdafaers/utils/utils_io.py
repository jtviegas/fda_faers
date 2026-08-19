"""I/O utility helpers."""

import logging
import urllib.request
from urllib.error import HTTPError

logger = logging.getLogger(__name__)

class UtilsIOError(Exception):
    """Exception raised for UtilsIO-related errors."""

class UtilsIO:
    """Utility methods for I/O-related operations."""

    @staticmethod
    def resource_exists(url: str) -> bool:
        """Return the resolved URL if the file exists (HTTP HEAD), else None."""
        logger.info(f"[resource_exists|in] ({url})")
        result: bool = False
        try:
            req = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(req) as response:
                result = (200 == response.code)
        except HTTPError as x:
            raise UtilsIOError(f"[resource_exists] failed request to: {url}") from x

        logger.info(f"[resource_exists|out] => {result}")
        return result

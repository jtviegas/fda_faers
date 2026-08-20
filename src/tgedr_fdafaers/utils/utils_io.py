"""I/O utility helpers."""

from collections.abc import Callable
import logging
from pathlib import Path
import tempfile
import shutil
from typing import ClassVar
import urllib.request
from urllib.error import HTTPError
from urllib.parse import urlparse
import zipfile

logger = logging.getLogger(__name__)

class UtilsIOError(Exception):
    """Exception raised for UtilsIO-related errors."""

class UtilsIO:
    """Utility methods for I/O-related operations."""

    __ALLOWED_SCHEMES: ClassVar[set[str]] = {"http", "https", "dbfs"}

    @staticmethod
    def tmp_dir() -> str:
        """Create and return the path to a temporary directory."""
        _folder = tempfile.TemporaryDirectory("+wb").name
        _path = Path(_folder)
        if not _path.exists():
            _path.mkdir(parents=True)
        return _folder

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

    @staticmethod
    def deflate_zip(file: str, target_folder: str, file_filter: Callable[[str], bool] = lambda f: True,  # noqa: ARG005
                    lower_filename: bool = False) -> list[str]:  # noqa: FBT001, FBT002
        """
        helper function to deflate a zip file
        """
        logger.debug(f"[deflate_zip|in] ({file}, {target_folder})")
        result: list[str] = []

        zipdata = zipfile.ZipFile(file)
        zipinfos = zipdata.infolist()

        # iterate through each file in the zip archive
        for zipinfo in zipinfos:
            zipfile_name = Path(zipinfo.filename).name
            if file_filter(zipfile_name):
                zipdata.extract(zipinfo.filename, path=target_folder)
                end_file = str(Path(target_folder) / zipinfo.filename)
                if lower_filename:
                    zipfile_folder = Path(zipinfo.filename).parent
                    end_file_normalized = str(Path(target_folder) / zipfile_folder / zipfile_name.lower())
                    shutil.move(end_file, end_file_normalized)
                    end_file = end_file_normalized
                result.append(end_file)

        logger.debug(f"[deflate_zip|out] => {result}")
        return result

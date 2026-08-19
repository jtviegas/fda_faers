"""Databricks Volume-backed FileStore implementation with a local-filesystem fallback."""

import fnmatch
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tgedr_dataops_ext.commons.utils_spark import UtilsSpark

from tgedr_fdafaers.file_store import FileStore, FileStoreError

logger = logging.getLogger(__name__)


@dataclass
class _FileInfo:
    """Mimics the Databricks FileInfo object returned by dbutils.fs.ls()."""

    path: str
    name: str
    size: int


class _LocalFs:
    """Local filesystem adapter implementing the same interface as ``dbutils.fs``.

    Enables VolumeFiles to operate on the local filesystem without Databricks.
    """

    def ls(self, path: str) -> list[_FileInfo]:
        """List files and directories at *path*, returning ``_FileInfo`` entries."""
        path = path.rstrip("/")
        if not Path(path).exists():
            raise FileNotFoundError(f"path does not exist: {path}")
        if Path(path).is_file():
            return [_FileInfo(path=path, name=Path(path).name, size=Path(path).stat().st_size)]
        entries: list[_FileInfo] = []
        for name in os.listdir(path):
            full = str(Path(path) / name)
            if Path(full).is_dir():
                entries.append(_FileInfo(path=full + "/", name=name, size=0))
            else:
                entries.append(_FileInfo(path=full, name=name, size=Path(full).stat().st_size))
        return entries

    def cp(self, src: str, dst: str, *, recurse: bool = False) -> bool:
        """Copy *src* to *dst*, creating parent directories as needed."""
        src = src.rstrip("/")
        dst = dst.rstrip("/")
        # keep the parameter to maintain signature compatibility; mark as used
        _ = recurse
        if not Path(src).exists():
            raise FileNotFoundError(f"source not found: {src}")
        dst_parent = Path(dst).parent
        dst_parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return True

    def rm(self, path: str, *, recurse: bool = False) -> bool:
        """Remove *path*. If *recurse* is True and *path* is a directory, remove it recursively."""
        path = path.rstrip("/")
        if not Path(path).exists():
            raise FileNotFoundError(f"path not found: {path}")
        if Path(path).is_dir() and recurse:
            shutil.rmtree(path)
        else:
            Path(path).unlink()
        return True

    def mkdirs(self, path: str) -> bool:
        """Create the directory tree at *path*, including intermediate parents."""
        Path(path.rstrip("/")).mkdir(parents=True, exist_ok=True)
        return True

class DatabricksVolumeFileStore(FileStore):
    """FileStore backed by a Databricks Unity Catalog Volume (or local filesystem for testing)."""

    CONFIG_KEY_LOCAL_FS = "local"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initializes the DatabricksVolumeFileStore instance.

        Args:
            config: Optional configuration dictionary.
        """
        FileStore.__init__(self, config=config)

        use_local_fs: bool = config.get(self.CONFIG_KEY_LOCAL_FS, False) if config else False

        if use_local_fs:
            self._fs = _LocalFs()
        else:
            from pyspark.dbutils import DBUtils # pyright: ignore[reportMissingImports]
            spark = UtilsSpark.get_spark_session()
            self._fs = DBUtils(spark).fs

    def __path_exists(self, path: str) -> bool:
        """Return True if *path* can be listed (exists in the filesystem)."""
        try:
            self._fs.ls(path)
            return True  # noqa: TRY300
        except Exception:  # noqa: BLE001
            return False

    def __is_file(self, path: str) -> bool:
        """Return True if *path* refers to a single file (not a directory)."""
        try:
            entries = self._fs.ls(path)
            # if ls returns exactly one entry whose path matches the queried path, it is a file
            if len(entries) == 1:
                entry_path = entries[0].path.rstrip("/")
                return entry_path == path.rstrip("/")
            return False  # noqa: TRY300
        except Exception:  # noqa: BLE001
            return False

    def list(self, key: str, **kwargs) -> list[str]:
        """List file paths under *key*, optionally filtered by a ``file_pattern`` glob."""
        logger.info(f"[list|in] ({key}, {kwargs})")

        file_pattern: str = kwargs.get("file_pattern", "*")

        try:
            entries = self._fs.ls(key)
        except Exception as ex:
            raise FileStoreError(f"[list] volume path does not exist or is not accessible: {key} - {ex}") from ex

        # entries returned by dbutils.fs.ls have .path and .size attributes
        # directories have paths ending with '/', files do not (or have size > 0)
        result: list[str] = sorted(
            [
                entry.path
                for entry in entries
                if not entry.path.endswith("/") and fnmatch.fnmatch(Path(entry.path).name, file_pattern)
            ]
        )

        logger.info(f"[list|out] => {len(result)} file(s)")
        return result

    def get(self, key: str, **kwargs) -> None:
        """Download/copy the file at *key* to the local path specified by ``target_url``."""
        logger.info(f"[get|in] ({key}, {kwargs})")

        if "target_url" not in kwargs:
            raise FileStoreError("[get] you must provide 'target_url' in kwargs")
        target_url: str = kwargs["target_url"]

        recursive: bool = kwargs.get("recursive", False)

        try:
            self._fs.cp(key, target_url, recurse=recursive)
        except Exception as ex:
            raise FileStoreError(f"[get] failed to download file: {key} to {target_url}") from ex

        logger.info("[get|out]")

    def delete(self, key: str, **kwargs) -> None:
        """Delete the file or directory at *key*. Pass ``recursive=True`` to remove directories."""
        logger.info(f"[delete|in] ({key}, {kwargs})")

        recursive: bool = kwargs.get("recursive", False)
        try:
            self._fs.rm(key, recurse=recursive)
        except Exception as ex:
            raise FileStoreError(f"[delete] failed to delete file: {key}") from ex

        logger.info("[delete|out]")

    def put(self, source: str, target: str) -> None:
        """Upload/copy a local file at *source* to the store at *target*."""
        logger.info(f"[put|in] ({source}, {target})")

        try:
            target_parent = str(target).rsplit("/", 1)[0]
            self._fs.mkdirs(target_parent)
            self._fs.cp(source, target)
        except Exception as ex:
            raise FileStoreError(f"[put] failed to copy {source} to: {target}") from ex

        logger.info("[put|out]")

"""Process and load corrected FAERS period files."""

from typing import Any
import logging
import os
from pathlib import Path

from tgedr_dataops_abs.etl4gh import Etl4GH
from tgedr_fdafaers.faers_files_corrections import FaersFilesCorrections
from tgedr_fdafaers.utils.utils_io import UtilsIO


logger = logging.getLogger(__name__)


class ProcessPeriodFiles(Etl4GH):
    """ETL workflow for extracting, correcting, and loading FAERS period files."""

    def __init__(self, configuration: dict[str, Any] | None = None) -> None:
        """Initialise the ETL with runtime configuration."""
        super().__init__(configuration=configuration)
        self._tmp_dir: str = UtilsIO.tmp_dir()
        self._output_dir: str = UtilsIO.tmp_dir()
        self._files_deflated: list[str] = []

    @Etl4GH.inject_configuration
    def extract(self, files: str) -> Any:
        """Extract relevant FAERS text files from the specified archives."""
        logger.info(f"[extract|in] ({files})")

        def zif_files_filter(file_name: str) -> bool:
            """Filter for deflating only the relevant FAERS text files."""
            return (
                str.lower(file_name).endswith(".txt")
                and str.lower(file_name)[:4] not in ["size", "stat"]
                and file_name[0] != "."
            )

        for file in [f.strip() for f in files.split(",")]:
            logger.info(f"[extract] processing file: {file}")
            self._files_deflated.extend(
                UtilsIO.deflate_zip(file, self._tmp_dir,
                    file_filter=zif_files_filter, lower_filename=True)
            )
        logger.info(f"[extract|out] => files_deflated: {len(self._files_deflated)}")

    def transform(self) -> Any:
        """Transform step for file status processing workflow."""
        logger.info("[transform|in]")
        # correct files
        if 0 < len(self._files_deflated):
            corrections = FaersFilesCorrections()
            for file in self._files_deflated:
                logger.info(f"[process] correcting file: {file}")
                corrections.process(context={"input_file": file, "output_folder": self._output_dir})
        logger.info("[transform|out]")

    def load(self) -> str:
        """Return the corrected file paths as a comma-separated string."""
        logger.info("[load|in]")
        new_files: list[str] = [
            str(Path(self._output_dir) / file) for file in os.listdir(self._output_dir)
        ]
        result: str = ",".join(new_files)
        logger.info(f"[load|out] => {result}")
        return result

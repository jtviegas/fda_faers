"""Process and load corrected FAERS period files."""

from typing import Any
import logging
import os
from pathlib import Path

from tgedr_dataops_abs.processor import Processor
from tgedr_fdafaers.faers_files_corrections import FaersFilesCorrections
from tgedr_fdafaers.utils.utils_io import UtilsIO


logger = logging.getLogger(__name__)


class ProcessPeriodFiles(Processor):
    """ETL workflow for extracting, correcting, and loading FAERS period files."""

    CONTEXT_KEY_FILES = "files"

    def process(self, context: dict[str, Any] | None = None) -> list[str] | None:
        """Process the given FAERS period files.

        Extracts the relevant text files from the provided zip files, applies
        the FAERS corrections, and returns the list of corrected output files.

        Args:
            context: Runtime context containing the ``files`` key with the list
                of period files to process.

        Returns:
            The list of corrected output file paths, or ``None`` if there are
            no files to process.

        Raises:
            ValueError: If the required ``files`` key is missing from context.
        """
        logger.info(f"[process|in] ({context})")

        if self.CONTEXT_KEY_FILES not in context:
            raise ValueError(f"[process] Missing required context key: {self.CONTEXT_KEY_FILES}")
        files: list[str] = context[self.CONTEXT_KEY_FILES]

        result: list[str] | None = None

        if 0 < len(files):
            tmp_dir: str = UtilsIO.tmp_dir()
            files_deflated: list[str] = []

            def zif_files_filter(file_name: str) -> bool:
                """Filter for deflating only the relevant FAERS text files."""
                return (
                    str.lower(file_name).endswith(".txt")
                    and str.lower(file_name)[:4] not in ["size", "stat"]
                    and file_name[0] != "."
                )

            for file in [f.strip() for f in files]:
                logger.info(f"[extract] processing file: {file}")
                files_deflated.extend(
                    UtilsIO.deflate_zip(file, tmp_dir, file_filter=zif_files_filter, lower_filename=True)
                )

            if 0 < len(files_deflated):
                output_dir: str = UtilsIO.tmp_dir()
                corrections = FaersFilesCorrections()
                for file in files_deflated:
                    logger.info(f"[process] correcting file: {file}")
                    corrections.process(context={"input_file": file, "output_folder": output_dir})

                result = [str(Path(output_dir) / file) for file in os.listdir(output_dir)]

        logger.info(f"[process|out] => {result}")
        return result

"""Module for fetching FAERS period files from configured sources."""

import os
from typing import Any
import logging
from tgedr_dataops_abs.processor import Processor
from pvprototypes_faers.files_handling.faers_file_source import FaersFileSource


logging.getLogger("py4j").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)


class PeriodFileFetcher(Processor):
    """Fetches period files from FAERS source based on provided periods and URL configuration."""

    CONTEXT_KEY_ZIP_FILES_URL = "zip_files_url"
    CONTEXT_KEY_PERIODS = "periods"

    def process(self, context: dict[str, Any] | None = None) -> list[str]:
        """Fetch period files from FAERS source based on configured periods.

        Args:
            context: Configuration dictionary containing 'zip_files_url' and 'periods'.

        Returns:
            List of downloaded period file names.

        Raises:
            Exception: If required context keys are missing.
        """
        logger.info(f"[process|in] ({context})")

        if not context or self.CONTEXT_KEY_ZIP_FILES_URL not in context:
            raise Exception(f"[process] configuration must include '{self.CONTEXT_KEY_ZIP_FILES_URL}'")
        if self.CONTEXT_KEY_PERIODS not in context:
            raise Exception(f"[process] configuration must include '{self.CONTEXT_KEY_PERIODS}'")

        zip_files_url: str = context[self.CONTEXT_KEY_ZIP_FILES_URL]
        periods: list[str] = context[self.CONTEXT_KEY_PERIODS]

        files_source = FaersFileSource()
        result = [
            os.path.basename(files_source.get(context={"period": period, "output_url": zip_files_url}))
            for period in periods
        ]
        logger.info(f"[process|out] => period files downloaded: {result}")
        return result

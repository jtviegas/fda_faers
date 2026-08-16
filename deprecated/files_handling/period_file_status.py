"""Module for handling FAERS period file status checks.

This module provides functionality to identify missing FAERS data files
by comparing available zip files with expected periods.
"""

import os
from typing import Any
import logging
from tgedr_dataops_abs.processor import Processor
from pvprototypes_faers.files_handling.faers_periods import FaersPeriods
from tgedr_dataops_ext.both.volume_files import NoSourceException, VolumeFiles


logger = logging.getLogger(__name__)


class PeriodFileStatus(Processor):
    """Processor for determining missing FAERS period files.

    This processor checks a volume path for available FAERS zip files and
    returns a list of periods with missing data files.
    """

    __ENV_KEY_PYSPARK_IS_LOCAL = "PYSPARK_IS_LOCAL"
    CONFIG_KEY_ZIP_FILES_URL = "zip_files_url"

    @property
    def _is_local_processing(self) -> bool:
        return os.getenv(self.__ENV_KEY_PYSPARK_IS_LOCAL) == "1"

    def process(self, context: dict[str, Any] | None = None) -> list[str]:
        """Identify missing FAERS period files.

        Args:
          context: Configuration dictionary containing 'zip_files_url' key with
            the path to the directory containing FAERS zip files.

        Returns:
          List of period strings that have missing data files.

        Raises:
          Exception: If context is missing or does not contain required configuration key.
        """
        logger.info(f"[process|in] ({context})")

        if not context or self.CONFIG_KEY_ZIP_FILES_URL not in context:
            raise Exception(f"[process] configuration must include '{self.CONFIG_KEY_ZIP_FILES_URL}'")

        zip_files_url = context[self.CONFIG_KEY_ZIP_FILES_URL]
        os.makedirs(zip_files_url, exist_ok=True)

        volume_config: dict[str, Any] = {}
        if self._is_local_processing:
            volume_config[VolumeFiles.CONFIG_KEY_USE_LOCAL_FS] = "true"

        volume_files = VolumeFiles(config=volume_config)
        try:
            files: list[str] = volume_files.list(
                {
                    VolumeFiles.CONTEXT_KEY_VOLUME_PATH: zip_files_url,
                    VolumeFiles.CONTEXT_KEY_FILE_PATTERN: "*.zip",
                }
            )
        except NoSourceException as nsx:
            logger.warning(f"[extract] volume path not found or inaccessible: {zip_files_url} - {nsx}")
            # When path is inaccessible, all periods are considered missing
            result = [str(x) for x in FaersPeriods.get_all_faers_periods()]
        else:
            all_periods: list[str] = [str(x) for x in FaersPeriods.get_all_faers_periods()]
            file_periods: list[str] = []
            for file_path in files:
                period = FaersPeriods.resolve_period(os.path.basename(file_path))
                if period is not None:
                    file_periods.append(str(period))

            result = [p for p in all_periods if p not in file_periods]
            result.sort()

        logger.info(f"[process|out] => missing periods: {result}")
        return result

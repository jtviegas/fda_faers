"""ETL module for managing FAERS file status and processing."""

from typing import Any
import logging
import os

from tgedr_dataops_ext.commons.etl_databricks import EtlDatabricks
from tgedr_dataops_ext.sink.catalog_file_sink import CatalogFileSink
from pvprototypes_faers.files_handling.faers_files_corrections import FaersFilesCorrections
from pvprototypes_faers.files_handling.period_file_deflater import PeriodFileDeflater
from pvprototypes_faers.files_handling.period_file_fetcher import PeriodFileFetcher
from pvprototypes_faers.files_handling.period_file_status import PeriodFileStatus


logger = logging.getLogger(__name__)


class FilesHandling(EtlDatabricks):
    """ETL class for managing FAERS file status and processing workflows."""

    CONFIG_KEY_ZIP_FILES_URL = "zip_files_url"
    CONFIG_KEY_MAX_PERIODS = "max_periods"
    CONFIG_KEY_TMP_URL = "tmp_url"
    CONFIG_KEY_OUTPUT_URL = "output_url"

    __DEFLATED_FILES_FOLDER = "deflated"
    __CORRECTED_FILES_FOLDER = "corrected"

    def __init__(self, configuration: dict[str, Any] | None = None) -> None:
        """Initialise the ETL with runtime configuration."""
        super().__init__(configuration=configuration)
        self._files_fetched: list[str] = []
        self._deflated_files_url: str | None = None
        self._corrected_files_url: str | None = None

    @EtlDatabricks.inject_configuration
    def extract(self, zip_files_url: str, max_periods: str) -> Any:
        """
        Lists files in the volume and resolves missing periods.
        """
        logger.info(f"[extract|in] ({zip_files_url}, {max_periods})")
        self._periods_missing = PeriodFileStatus().process(
            context={
                PeriodFileStatus.CONFIG_KEY_ZIP_FILES_URL: zip_files_url,
            }
        )
        max_periods_int = int(max_periods)
        periods_to_process: list[str] = self._periods_missing[:max_periods_int]
        logger.info(f"[load] periods_to_process: {periods_to_process}")

        # fetch zip files for missing periods (limited by max_periods)
        self._files_fetched: list[str] = PeriodFileFetcher().process(
            context={
                PeriodFileFetcher.CONTEXT_KEY_ZIP_FILES_URL: zip_files_url,
                PeriodFileFetcher.CONTEXT_KEY_PERIODS: periods_to_process,
            }
        )
        logger.info(f"[extract|out] => files_fetched: {len(self._files_fetched)}")

    @EtlDatabricks.inject_configuration
    def transform(self, zip_files_url: str, tmp_url: str) -> Any:
        """Transform step for file status processing workflow."""
        logger.info(f"[transform|in] ({zip_files_url}, {tmp_url})")

        self._deflated_files_url = os.path.join(tmp_url, self.__DEFLATED_FILES_FOLDER)
        self._corrected_files_url = os.path.join(tmp_url, self.__CORRECTED_FILES_FOLDER)

        deflated_files: list[str] = PeriodFileDeflater(
            {
                PeriodFileDeflater.CONFIG_KEY_INPUT_FOLDER: zip_files_url,
                PeriodFileDeflater.CONFIG_KEY_OUTPUT_FOLDER: self._deflated_files_url,
            }
        ).process(
            context={
                PeriodFileDeflater.CONTEXT_KEY_FILE_FILTER: self._files_fetched,
            }
        )
        logger.info(f"[transform] deflated files: {deflated_files}")

        # correct files
        if 0 < len(deflated_files):
            corrections = FaersFilesCorrections()
            for file in deflated_files:
                logger.info(f"[process] correcting file: {file}")
                corrections.process(context={"input_file": file, "output_folder": self._corrected_files_url})

        logger.info("[transform|out]")

    @EtlDatabricks.inject_configuration
    def load(self, tmp_url: str, output_url: str) -> dict[str, Any]:
        """Move processed files to final output location."""
        logger.info(f"[load|in] ({tmp_url}, {output_url})")

        # move to final location
        sink: CatalogFileSink = CatalogFileSink()
        new_files: list[str] = []
        for file in [f for f in os.listdir(self._corrected_files_url)]:
            logger.info(f"[process] moving file: {file} to {output_url}")
            source_file = os.path.join(self._corrected_files_url, file)  # pyright: ignore[reportArgumentType, reportCallIssue]
            target_file = os.path.join(output_url, file)
            sink.put(context={"source": source_file, "target": target_file})
            new_files.append(target_file)

        result: dict[str, Any] = {"new_files": new_files}
        logger.info(f"[load|out] => {result}")
        return result

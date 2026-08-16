"""Resolve the FDA download URL for a given FAERS/AERS period."""

import logging

from pvprototypes_faers.files_handling.faers_periods import FaersPeriod, FaersPeriods

logger = logging.getLogger(__name__)


class FaersFileUrl:
    """
    Resolves the faers file path for a given period, based on the period and the data format (FAERS or AERS).
    """

    __URL_FILE_PATTERN = "https://fis.fda.gov/content/Exports/{faers_prefix}_ascii_{year}{quarter_midfix}{quarter}.zip"

    def get_url(self, period: FaersPeriod) -> str:
        """
        based on the period faers file names have a different format, as there were changes
        in the last quarter of 2012, back then the files were called `aers` and become `faers`, so
        we need to derive the file names differently depending on the period
        """
        logger.info(f"[get_url|in] ({period})")

        faers_prefix = "aers"
        quarter_midfix = "q"
        if FaersPeriods.is_faers_period(period):
            faers_prefix = "faers"
            quarter_midfix = "Q"

        result = self.__URL_FILE_PATTERN.format(
            faers_prefix=faers_prefix,
            year=period.year,
            quarter_midfix=quarter_midfix,
            quarter=period.quarter,
        )

        logger.info(f"[get_url|out] =>{result}")
        return result

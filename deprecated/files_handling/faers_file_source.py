"""Source adapter that checks availability and downloads FAERS files from the FDA."""

import os
from pathlib import Path
from typing import Any, Optional, Dict
import urllib.request
from urllib.error import HTTPError
import logging
from tgedr_dataops_abs.source import Source, SourceException

from pvprototypes_faers.files_handling.faers_file_url import FaersFileUrl
from pvprototypes_faers.files_handling.faers_periods import FaersPeriod


logger = logging.getLogger(__name__)


class FaersFileSource(Source):
    """Check availability and download FAERS quarterly zip files from the FDA endpoint."""

    CONTEXT_KEY_OUTPUT_URL = "output_url"
    CONTEXT_KEY_PERIOD = "period"
    OUTPUT_FILE_PATTERN = "{year}q{quarter}.zip"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialise the source adapter."""
        super().__init__(config=config)

    def __get_url(self, context: dict[str, Any]) -> str:
        period_param: str = context[self.CONTEXT_KEY_PERIOD]
        period: FaersPeriod = FaersPeriod.from_str(
            period_param
        )  # just to validate the provided period, if not valid it will raise an exception
        return FaersFileUrl().get_url(
            period
        )  # just to validate the provided period, if not valid it will raise an exception

    def list(self, context: dict[str, Any] | None = None) -> Any:
        """Return the resolved URL if the file exists (HTTP HEAD), else None."""
        logger.info(f"[list|in] ({context})")
        result: str | None = None
        if not context or self.CONTEXT_KEY_PERIOD not in context:
            raise SourceException(f"[list] you must provide context for {self.CONTEXT_KEY_PERIOD}")

        period: FaersPeriod = FaersPeriod.from_str(
            context[self.CONTEXT_KEY_PERIOD]
        )  # just to validate the provided period, if not valid it will raise an exception
        url: str = FaersFileUrl().get_url(period)
        logger.info(f"[list] checking url: {url}")
        try:
            req = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(req) as response:
                if 200 == response.code:
                    result = response.url
        except HTTPError as x:
            raise SourceException(f"[list] failed request to: {url} - {x}")

        logger.info(f"[list|out] => {result}")
        return result

    def get(self, context: Optional[Dict[str, Any]] = None) -> str:
        """Download the FAERS zip file for the given period to the output URL."""
        logger.info(f"[get|in] ({context})")

        if not context or self.CONTEXT_KEY_PERIOD not in context:
            raise SourceException(f"[get] you must provide context for {self.CONTEXT_KEY_PERIOD}")
        if not context or self.CONTEXT_KEY_OUTPUT_URL not in context:
            raise SourceException(f"[get] you must provide context for {self.CONTEXT_KEY_OUTPUT_URL}")

        output_url: str = context[self.CONTEXT_KEY_OUTPUT_URL]
        os.makedirs(output_url, exist_ok=True)

        period: FaersPeriod = FaersPeriod.from_str(
            context[self.CONTEXT_KEY_PERIOD]
        )  # just to validate the provided period, if not valid it will raise an exception
        url: str = FaersFileUrl().get_url(period)
        target_url: str = str(
            Path(output_url).joinpath(self.OUTPUT_FILE_PATTERN.format(year=period.year, quarter=period.quarter))
        )
        logger.info(f"[get] retrieving file: {target_url} from url: {url} ")

        try:
            urllib.request.urlretrieve(url, target_url)
        except HTTPError as x:
            if 404 != x.code:
                raise SourceException(f"[get] failed request to: {url}")
            logger.error(f"[get] ooppss: {x}", exc_info=x)
            logger.warning(f"[get] url not found: {url}")
        except Exception as ex:
            logger.error(f"[get] ooppss: {ex}", exc_info=ex)
            raise SourceException(f"[get] failed request to: {url}")

        logger.info(f"[get|out] => {target_url}")
        return target_url

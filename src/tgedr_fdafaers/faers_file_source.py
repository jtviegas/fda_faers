"""Source adapter that checks availability and downloads FAERS files from the FDA."""

import logging
import urllib.request
from pathlib import Path
from typing import Any, ClassVar
from urllib.error import HTTPError
from urllib.parse import urlparse

from tgedr_dataops_abs.source import Source, SourceException

from tgedr_fdafaers.utils.faers_period import FaersPeriod, UtilsFaersPeriod
from tgedr_fdafaers.utils.utils_io import UtilsIO

logger = logging.getLogger(__name__)


class FaersFileSource(Source):
    """Check availability and download FAERS quarterly zip files from the FDA endpoint."""

    CONTEXT_KEY_OUTPUT_URL = "output_url"
    CONTEXT_KEY_PERIOD = "period"
    OUTPUT_FILE_PATTERN = "{year}q{quarter}.zip"
    __ALLOWED_SCHEMES: ClassVar[set[str]] = {"http", "https", "dbfs"}

    def __get_period_url(self, context: dict[str, Any]) -> tuple[FaersPeriod, str]:
        """Extract and validate the period from *context*, returning it and the resolved download URL."""
        logger.info(f"[__get_period_url|in] ({context})")
        if not context or self.CONTEXT_KEY_PERIOD not in context:
            msg = f"[__get_period_url] you must provide context for {self.CONTEXT_KEY_PERIOD}"
            raise SourceException(msg)
        period: FaersPeriod = FaersPeriod.from_str(
                    context[self.CONTEXT_KEY_PERIOD]
                )  # just to validate the provided period, if not valid it will raise an exception
        result = UtilsFaersPeriod.get_url(period)
        logger.info(f"[__get_period_url|out] => {result}")
        return period, result

    def list(self, context: dict[str, Any]) -> str | None:
        """Return the resolved URL for the provided period if the file exists (HTTP HEAD), else None."""
        logger.info(f"[list|in] ({context})")
        result: str | None = None

        _, url = self.__get_period_url(context)
        if UtilsIO.resource_exists(url):  # just to validate the provided period, if not valid it will raise an exception
            result = url

        logger.info(f"[list|out] => {result}")
        return result

    def get(self, context: dict[str, Any]) -> str:
        """Download the FAERS zip file for the given period to the output URL."""
        logger.info(f"[get|in] ({context})")

        if not context or self.CONTEXT_KEY_OUTPUT_URL not in context:
            msg: str = f"[get] you must provide context for {self.CONTEXT_KEY_OUTPUT_URL}"
            raise SourceException(msg)
        output_url: str = context[self.CONTEXT_KEY_OUTPUT_URL]
        Path(output_url).mkdir(parents=True, exist_ok=True)

        period, url = self.__get_period_url(context)
        target_url: str = str(
            Path(output_url).joinpath(self.OUTPUT_FILE_PATTERN.format(year=period.year, quarter=period.quarter))
        )
        logger.info(f"[get] retrieving file: {target_url} from url: {url} ")

        scheme = urlparse(target_url).scheme
        if 0 < len(scheme) and scheme not in self.__ALLOWED_SCHEMES:
            raise SourceException(f"[get] unsupported URL scheme: {scheme} - {target_url}")

        try:
            urllib.request.urlretrieve(url, target_url)   # nosec B310
        except HTTPError as x:
            if 404 != x.code:
                raise SourceException(f"[get] failed request to: {url}") from x
            logger.warning(f"[get] url not found: {url}")
        except Exception as ex:
            raise SourceException(f"[get] failed request to: {url}") from ex

        logger.info(f"[get|out] => {target_url}")
        return target_url

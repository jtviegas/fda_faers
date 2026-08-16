"""FAERS period representation and period-range utilities."""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional
import re
import logging


logger = logging.getLogger(__name__)


@dataclass
class FaersPeriod:
    """
    data class depicting the concept of an FAERS period
    """

    year: int
    quarter: int

    def __lt__(self, other):
        """Return True if this period is strictly before *other*."""
        return (self.year == other.year and self.quarter < other.quarter) or (self.year < other.year)

    def __le__(self, other):
        """Return True if this period is before or equal to *other*."""
        return (self.year == other.year and self.quarter <= other.quarter) or (self.year < other.year)

    def __gt__(self, other):
        """Return True if this period is strictly after *other*."""
        return (self.year == other.year and self.quarter > other.quarter) or (self.year > other.year)

    def __ge__(self, other):
        """Return True if this period is after or equal to *other*."""
        return (self.year == other.year and self.quarter >= other.quarter) or (self.year > other.year)

    def __eq__(self, other):
        """Return True if both periods represent the same year and quarter."""
        return self.year == other.year and self.quarter == other.quarter

    def __ne__(self, other):
        """Return True if the periods differ in year or quarter."""
        return self.year != other.year or self.quarter != other.quarter

    def __str__(self) -> str:
        """Return the short string form, e.g. ``'21q1'``."""
        return f"{str(self.year)[2:]}q{self.quarter}"

    @staticmethod
    def from_str(period: str) -> "FaersPeriod":
        """
        converts an FaersPeriod from its string representation back to its structured representation
        """
        elements = period.split("q")
        # DISCLAIMER: we assume this code will fail in 75 years from now
        # --->>> RUN UNIT TESTS and it will surface ( ...as if... :-) )
        year = int(("20" + elements[0]))
        quarter = int(elements[1])

        return FaersPeriod(year, quarter)


class FaersPeriods:
    """
    handy class comprising static methods for FaersPeriod operations
    """

    AERS_END_PERIOD = FaersPeriod(2012, 3)
    AERS_START_PERIOD = FaersPeriod(2004, 1)

    @staticmethod
    def get_current_period() -> FaersPeriod:
        """
        returns the current FAERS period based on the current UTC date
        """
        logger.debug("[get_current_period|in]")
        now = datetime.now(timezone.utc)
        result = FaersPeriod(now.year, (int((now.month - 1) / 3) + 1) - 1)
        logger.debug(f"[get_current_period|out] => {result}")
        return result

    @staticmethod
    def next_period(period: FaersPeriod) -> FaersPeriod:
        """Return the period immediately following *period*."""
        logger.debug(f"[next_period|in] ({period})")

        year: int = period.year
        quarter: int = period.quarter
        if 4 == quarter:
            quarter = 1
            year += 1
        else:
            quarter += 1

        result = FaersPeriod(year, quarter)
        logger.debug(f"[next_period|out] => {result}")
        return result

    @staticmethod
    def previous_period(period: FaersPeriod) -> FaersPeriod:
        """Return the period immediately preceding *period*."""
        logger.debug(f"[previous_period|in] ({period})")

        year: int = period.year
        quarter: int = period.quarter
        if 1 == quarter:
            quarter = 4
            year -= 1
        else:
            quarter -= 1

        result = FaersPeriod(year, quarter)
        logger.debug(f"[previous_period|out] => {result}")
        return result

    @staticmethod
    def is_faers_period(period: FaersPeriod) -> bool:
        """
        checks if the provided period is valid, as in it is after the beginning of FAERS data format availability

        Context: based on the period faers file names have a different format, as there were changes
        in the last quarter of 2012, back then the files were called `aers` and become `faers`.

        """
        logger.debug(f"[is_faers_processing|in] ({period})")
        result = period > FaersPeriods.AERS_END_PERIOD
        logger.debug(f"[is_faers_processing|out] => {result}")
        return result

    @staticmethod
    def get_all_faers_periods() -> List[FaersPeriod]:
        """
        computes the overall list of periods since ever
        """
        logger.debug("[get_all_faers_periods|in]")
        result = FaersPeriods.get_faers_periods() + FaersPeriods.get_aers_periods()
        logger.debug(f"[get_all_faers_periods|out] => {result}")
        return result

    @staticmethod
    def get_faers_periods() -> List[FaersPeriod]:
        """
        computes the overall list of periods since the beginning of FAERS data format availability
        """
        logger.debug("[get_faers_periods|in]")

        current_period: FaersPeriod = FaersPeriods.get_current_period()

        period = FaersPeriods.next_period(FaersPeriods.AERS_END_PERIOD)

        result = []
        while period < current_period:
            result.append(period)
            period = FaersPeriods.next_period(period)

        logger.debug(f"[get_faers_periods|out] => {result}")
        return result

    @staticmethod
    def get_aers_periods() -> List[FaersPeriod]:
        """
        computes the overall list of periods since the beginning of AERS data format availability
        until the beginning of FAERS data format availability
        """
        logger.debug("[get_aers_periods|in]")

        period = FaersPeriods.AERS_START_PERIOD

        result = []
        while period <= FaersPeriods.AERS_END_PERIOD:
            result.append(period)
            period = FaersPeriods.next_period(period)

        logger.debug(f"[get_aers_periods|out] => {result}")
        return result

    @staticmethod
    def resolve_period(basename: str) -> Optional[FaersPeriod]:
        """
        Extracts a FaersPeriod from a filename basename.

        Supports both AERS (e.g. aers_ascii_2010q2.zip) and FAERS (e.g. faers_ascii_2023Q1.zip) formats.
        Returns None if no period can be resolved.
        """
        pattern = re.compile(r"(\d{4})[qQ](\d)")
        match = pattern.search(basename)
        if not match:
            return None
        year = int(match.group(1))
        quarter = int(match.group(2))
        return FaersPeriod(year, quarter)

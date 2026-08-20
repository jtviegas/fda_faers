"""Immutable constants for the FAERS data model: schemas, columns, partitions and comments."""

from dataclasses import dataclass
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Constants:
    """Immutable constants for FAERS data configuration."""

    CSV_DELIMITER: str = "$"
    TABLES: Tuple[str, ...] = ("demo", "drug", "indi", "outc", "reac", "rpsr", "ther")

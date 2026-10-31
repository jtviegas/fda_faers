
"""Base utilities for processing FAERS datasets."""

from typing import Any
from collections.abc import Sequence
from datasets import Dataset
from tgedr_dataops_abs.processor import Processor, ProcessorException

class FaersProcessor(Processor):
    """Base class for FAERS processors."""

    def _require_columns(self, dataset: Dataset, table: str, columns: set[str]) -> None:
        missing = columns.difference(dataset.column_names)
        if missing:
            raise ProcessorException(f"{table} is missing required columns: {', '.join(sorted(missing))}")

    def _version_key(self, version: Any) -> tuple[int, int]:
        return (0, -1) if version is None else (1, int(version))


    def _unique_values(self, rows: Sequence[dict[str, Any]], column: str) -> list[Any]:
        values = []
        seen = set()
        for row in rows:
            value = row.get(column)
            if value is not None and value not in seen:
                values.append(value)
                seen.add(value)
        return values

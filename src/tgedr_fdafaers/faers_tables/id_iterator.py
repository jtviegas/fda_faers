"""Id iterator processor for splitting lists of ids into batches.

This module provides:
- IdIterator: a Processor that yields batches of unique ids from a context.
"""

import logging
from collections.abc import Iterator
from typing import Any
from tgedr_dataops_abs.processor import Processor, ProcessorException


logger = logging.getLogger(__name__)


class IdIterator(Processor):
    """Splits a list of ids into batches, yielding each batch as a set of unique ids."""

    CONTEXT_KEY_IDS = "ids"
    CONTEXT_KEY_BATCH_SIZE = "batch_size"
    DEFAULT_BATCH_SIZE = 1000_000

    def process(self, context: dict[str, Any] | None = None) -> Iterator[set[int]]:
        """Yield batches of unique ids from the provided context.

        Args:
            context: Processing context. Must contain ``ids`` (a list of integers)
                and may contain ``batch_size`` (a positive integer). If ``batch_size``
                is not provided, :attr:`DEFAULT_BATCH_SIZE` is used.

        Yields:
            Sets of unique ids, each no larger than the batch size.

        Raises:
            ProcessorException: If ``ids`` is missing or not a list of integers, or
                if ``batch_size`` is not a positive integer.
        """
        logger.info("[process|in]")

        if not context or self.CONTEXT_KEY_IDS not in context:
            raise ProcessorException(f"{self.CONTEXT_KEY_IDS} must be provided in context")
        ids: list[int] = context[self.CONTEXT_KEY_IDS]
        if not isinstance(ids, list) or not all(isinstance(i, int) for i in ids):
            raise ProcessorException(f"{self.CONTEXT_KEY_IDS} must be a list of integers")

        batch_size: int = context.get(self.CONTEXT_KEY_BATCH_SIZE, self.DEFAULT_BATCH_SIZE)
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ProcessorException("batch_size must be a positive integer")

        for offset in range(0, len(ids), batch_size):
            batch_ids = ids[offset : offset + batch_size]
            yield set(batch_ids)

        logger.info("[process|out]")

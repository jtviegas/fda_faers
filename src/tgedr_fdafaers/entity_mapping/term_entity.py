"""Processor that parses entities from a term column and explodes them into individual rows.

This module provides:
- TermEntity: a Processor that parses entities from a term column and produces a
  dataframe with one entity per row.
"""

import logging
from typing import Any

import pandas as pd

from tgedr_dataops_abs.processor import Processor, ProcessorException
from tgedr_fdafaers.nlp.entity_parsing import parse_entities

logger = logging.getLogger(__name__)


class TermEntity(Processor):
    """Processor that parses entities from a term column and explodes them into individual rows."""

    DEFAULT_TERM_COLUMN = "term"
    DEFAULT_ENTITY_COLUMN = "entity"
    CONTEXT_KEY_DF = "dataframe"
    CONTEXT_KEY_TERM_COLUMN = "term_column"
    CONTEXT_KEY_ENTITY_COLUMN = "entity_column"

    def process(self, context: dict[str, Any] | None = None) -> pd.DataFrame:
        """Parse entities from the term column and produce a dataframe with one entity per row.

        Args:
            context: processing context; must contain the dataframe under the
                ``dataframe`` key, and may optionally specify the term and entity
                column names.

        Returns:
            A dataframe with the parsed entities exploded into individual rows.

        Raises:
            ProcessorException: if the context is missing or does not contain a
                dataframe.
        """
        logger.info(f"[process|in] ({context})")
        if not context or self.CONTEXT_KEY_DF not in context:
            raise ProcessorException(f"{self.CONTEXT_KEY_DF} must be provided in context")
        df: pd.DataFrame = context[self.CONTEXT_KEY_DF]

        term_column = context.get(self.CONTEXT_KEY_TERM_COLUMN, self.DEFAULT_TERM_COLUMN)
        entity_column = context.get(self.CONTEXT_KEY_ENTITY_COLUMN, self.DEFAULT_ENTITY_COLUMN)

        result: pd.DataFrame = (
            df.assign(parsed_entities=df[term_column].apply(parse_entities))
            .explode("parsed_entities", ignore_index=True)
            .rename(columns={"parsed_entities": entity_column})
            .drop_duplicates()
        )

        logger.info(f"[process|out] => {result.shape}")
        return result

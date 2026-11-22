"""Build and persist mappings from FAERS terms to ATC codes."""

from typing import Any
import logging
from pyspark.sql import DataFrame
from tgedr_dataops_abs.processor import Processor


logger = logging.getLogger(__name__)


class TermAtcMapping(Processor):
    """Build and persist mappings from FAERS terms to ATC codes."""

    CONTEXT_KEY_TERM_ENTITY_DF = "term_entity_dataset"
    CONTEXT_KEY_ENTITY_ATC_DF = "entity_atc_dataset"
    CONTEXT_KEY_ENTITY_COLUMN = "entity_column"

    __DEFAULT_ENTITY_COLUMN = "entity"

    def process(self, context: dict[str, Any] | None = None) -> DataFrame | None:
        """Build and persist mappings from FAERS terms to ATC codes.

        Args:
            context: Dictionary containing required keys:
                - term_entity_dataset: DataFrame with term-entity mappings
                - entity_atc_dataset: DataFrame with entity-ATC mappings
                - entity_column (optional): Column name for entity join (default: "entity")

        Returns:
            DataFrame with joined term-entity-ATC mappings or None if inputs are empty
        """
        logger.info(f"[process|in] ({context})")

        result: DataFrame | None = None

        if self.CONTEXT_KEY_TERM_ENTITY_DF not in context:
            raise ValueError(f"[process] Missing required context key: {self.CONTEXT_KEY_TERM_ENTITY_DF}")
        df_term_entity: DataFrame = context[self.CONTEXT_KEY_TERM_ENTITY_DF]

        if self.CONTEXT_KEY_ENTITY_ATC_DF not in context:
            raise ValueError(f"[process] Missing required context key: {self.CONTEXT_KEY_ENTITY_ATC_DF}")
        df_entity_atc: DataFrame = context[self.CONTEXT_KEY_ENTITY_ATC_DF]

        if not (df_entity_atc is None or df_entity_atc.rdd.isEmpty()) and not (
            df_term_entity is None or df_term_entity.rdd.isEmpty()
        ):
            entity_column = context.get(self.CONTEXT_KEY_ENTITY_COLUMN, self.__DEFAULT_ENTITY_COLUMN)
            result = df_term_entity.join(df_entity_atc, on=entity_column, how="inner")

        logger.info(f"[process|out] => {result}")
        return result

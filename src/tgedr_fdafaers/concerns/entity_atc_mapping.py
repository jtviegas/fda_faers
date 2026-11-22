"""ETL step that maps entities to ATC codes.

This module provides:
- EntityAtcMapping: an ETL step that extracts unmapped entities from the silver
  dataset, maps them to ATC codes using the NLM mapping, and loads the results
  back into the silver entity_atc table.
"""

from typing import Any
import logging
from pyspark.sql import DataFrame

from tgedr_dataops_abs.processor import Processor
from tgedr_fdafaers.atc_mapping.nlm_mapping import NLMMapping


logger = logging.getLogger(__name__)


class EntityAtcMapping(Processor):
    """ETL step that maps unmapped entities to ATC codes using the NLM mapping."""

    CONTEXT_KEY_TERM_ENTITY_DF = "term_entity_dataset"
    CONTEXT_KEY_ENTITY_ATC_DF = "entity_atc_dataset"
    CONTEXT_KEY_SAMPLE_SIZE = "sample_size"
    CONTEXT_KEY_ATC_COLUMN = "atc_column"
    CONTEXT_KEY_ENTITY_COLUMN = "entity_column"

    __DEFAULT_SAMPLE_SIZE = 50000
    __DEFAULT_ATC_COLUMN = "atc_codes"
    __DEFAULT_ENTITY_COLUMN = "entity"

    def process(self, context: dict[str, Any] | None = None) -> DataFrame | None:
        """Process the entity to ATC mapping ETL step.

        Args:
            context: Dictionary containing the required context keys:
                - term_entity_dataset: DataFrame with term entities
                - entity_atc_dataset: DataFrame with existing entity-ATC mappings
                - sample_size: Maximum number of entities to map (default: 50000)
                - atc_column: Name of the ATC column (default: "atc_codes")
                - entity_column: Name of the entity column (default: "entity")

        Returns:
            DataFrame with updated entity-ATC mappings or None if no mapping was performed.
        """
        logger.info(f"[process|in] ({context})")

        result: DataFrame | None = None

        if self.CONTEXT_KEY_TERM_ENTITY_DF not in context:
            raise ValueError(f"[process] Missing required context key: {self.CONTEXT_KEY_TERM_ENTITY_DF}")
        df_term_entity: DataFrame = context[self.CONTEXT_KEY_TERM_ENTITY_DF]

        if self.CONTEXT_KEY_ENTITY_ATC_DF not in context:
            raise ValueError(f"[process] Missing required context key: {self.CONTEXT_KEY_ENTITY_ATC_DF}")
        df_entity_atc: DataFrame = context[self.CONTEXT_KEY_ENTITY_ATC_DF]

        atc_column = context.get(self.CONTEXT_KEY_ATC_COLUMN, self.__DEFAULT_ATC_COLUMN)
        entity_column = context.get(self.CONTEXT_KEY_ENTITY_COLUMN, self.__DEFAULT_ENTITY_COLUMN)
        sample_size: int = int(context.get(self.CONTEXT_KEY_SAMPLE_SIZE, self.__DEFAULT_SAMPLE_SIZE))

        df_entities_to_map = self._find_entities_to_map(df_term_entity, df_entity_atc, entity_column, atc_column)
        n_entities: int = 0 if df_entities_to_map is None else df_entities_to_map.count()

        result = df_entity_atc
        if 0 < n_entities:
            fraction: float = min(1.0, sample_size / n_entities)
            df_sample = df_entities_to_map.sample(fraction=fraction)
            atc_mapper = NLMMapping()
            df_mapped = atc_mapper.decorate_pyspark(df=df_sample)
            if df_mapped is not None and not df_mapped.rdd.isEmpty():
                df_mapped = df_mapped.filter(f"{atc_column} IS NOT NULL").distinct()
                result = df_entity_atc.unionByName(df_mapped) if df_entity_atc is not None else df_mapped
            else:
                logger.info("[process] no new mapped entities")

        logger.info(f"[process|out] => {result}")
        return result

    def _find_entities_to_map(
        self, df_term_entity: DataFrame, df_entity_atc: DataFrame | None, entity_column: str, atc_codes_column: str
    ) -> DataFrame:
        logger.info(
            f"[_find_entities_to_map|in] ({df_term_entity}, {df_entity_atc}, {entity_column}, {atc_codes_column})"
        )
        df_mapped_entities: DataFrame | None = None
        if df_entity_atc is not None and not df_entity_atc.rdd.isEmpty():
            df_mapped_entities = (
                df_entity_atc.select(entity_column, atc_codes_column)
                .filter(f"{atc_codes_column} IS NOT NULL")
                .select(entity_column)
                .distinct()
            )

        if df_mapped_entities is not None and not df_mapped_entities.rdd.isEmpty():
            df_entities_to_map = (
                df_term_entity.select(entity_column)
                .distinct()
                .join(df_mapped_entities, on=entity_column, how="left_anti")
                .select(entity_column)
                .distinct()
            )
        else:
            df_entities_to_map = df_term_entity.select(entity_column).distinct()
        logger.info(f"[_find_entities_to_map|out] => {df_entities_to_map}")
        return df_entities_to_map

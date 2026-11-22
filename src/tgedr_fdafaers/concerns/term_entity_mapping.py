"""ETL step that maps drug terms to entities.

This module provides:
- TermEntityMapping: ETL step that extracts unmapped drug terms from bronze,
  maps them to entities, and loads the resulting mappings into silver.
"""

from typing import Any
import logging
import random
from pyspark.sql import DataFrame
import pyspark.sql.functions as F  # noqa: N812
from pyspark.sql.types import ArrayType, StringType
from tgedr_dataops_abs.processor import Processor
from tgedr_dataops_ext.commons.utils_spark import UtilsSpark
from tgedr_fdafaers.nlp.entity_parsing import parse_entities


logging.getLogger("great_expectations._docs_decorators").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


class TermEntityMapping(Processor):
    """ETL step that maps drug terms to entities."""

    CONTEXT_KEY_DRUG_DF = "drug_dataset"
    CONTEXT_KEY_TERM_ENTITY_DF = "term_entity_dataset"
    CONTEXT_KEY_SAMPLE_SIZE = "sample_size"
    CONTEXT_KEY_TERM_COLUMN = "term_column"
    CONTEXT_KEY_ENTITY_COLUMN = "entity_column"

    __DEFAULT_SAMPLE_SIZE = 50000
    __DEFAULT_TERM_COLUMN = "term"
    __DEFAULT_ENTITY_COLUMN = "entity"

    def process(self, context: dict[str, Any] | None = None) -> DataFrame:
        """Process the term-entity mapping ETL step.

        Args:
            context: Dictionary containing required keys:
                - drug_dataset: DataFrame with drug data
                - term_entity_dataset: DataFrame with already mapped terms
                Optional keys:
                - sample_size: Number of terms to sample for mapping (default: 50000)
                - term_column: Name of the term column (default: "term")
                - entity_column: Name of the entity column (default: "entity")

        Returns:
            DataFrame with new term-entity mappings or None if no new mappings found
        """
        logger.info(f"[process|in] ({context})")

        result: DataFrame | None = None

        if self.CONTEXT_KEY_DRUG_DF not in context:
            raise ValueError(f"[process] Missing required context key: {self.CONTEXT_KEY_DRUG_DF}")
        df_drug: DataFrame = context[self.CONTEXT_KEY_DRUG_DF]
        if self.CONTEXT_KEY_TERM_ENTITY_DF not in context:
            raise ValueError(f"[process] Missing required context key: {self.CONTEXT_KEY_TERM_ENTITY_DF}")
        df_term_entity: DataFrame = context[self.CONTEXT_KEY_TERM_ENTITY_DF]

        term_column = context.get(self.CONTEXT_KEY_TERM_COLUMN, self.__DEFAULT_TERM_COLUMN)
        entity_column = context.get(self.CONTEXT_KEY_ENTITY_COLUMN, self.__DEFAULT_ENTITY_COLUMN)
        sample_size: int = int(context.get(self.CONTEXT_KEY_SAMPLE_SIZE, self.__DEFAULT_SAMPLE_SIZE))

        terms = [row[0] for row in df_drug.select(term_column).distinct().collect()]
        logger.info(f"[process] terms count: {len(terms)}")
        mapped = (
            [
                row[0]
                for row in df_term_entity.filter(F.col(entity_column).isNotNull())
                .select(term_column)
                .distinct()
                .collect()
            ]
            if df_term_entity is not None
            else []
        )
        logger.info(f"[process] mapped terms count: {len(mapped)}")
        terms_to_map: list[str] = list(set(terms) - set(mapped))
        logger.info(f"[process] terms to map count: {len(terms_to_map)}")

        if 0 < len(terms_to_map):
            sample_terms: list[str] = random.sample(terms_to_map, min(sample_size, len(terms_to_map)))
            logger.info(f"[process] sample terms count: {len(sample_terms)}")

            df_new_mapped_terms: DataFrame = self._process_spark(sample_terms, term_column, entity_column)

            if df_new_mapped_terms is not None and not df_new_mapped_terms.rdd.isEmpty():
                if df_term_entity is not None:
                    result = df_term_entity.unionByName(df_new_mapped_terms)
                else:
                    result = df_new_mapped_terms
            else:
                logger.info("[process] no new mapped terms")
                result = df_term_entity

        logger.info(f"[process|out] => {result}")
        return result

    def _process_spark(self, terms: list[str], term_column: str, entity_column: str) -> DataFrame:
        logger.info(f"[_process_spark|in] ({terms}, {term_column}, {entity_column})")
        df_sample_terms: DataFrame = UtilsSpark.get_spark_session().createDataFrame(
            [[entry] for entry in terms], [term_column]
        )
        parse_entities_udf = F.udf(parse_entities, ArrayType(StringType()))

        result: DataFrame = (
            df_sample_terms.withColumn("parsed_entities", parse_entities_udf(F.col(term_column)))
            .withColumn(entity_column, F.explode_outer(F.col("parsed_entities")))
            .drop("parsed_entities")
            .dropDuplicates()
            .filter(F.col(entity_column).isNotNull())
        )
        logger.info(f"[_process_spark|out] => {result}")
        return result

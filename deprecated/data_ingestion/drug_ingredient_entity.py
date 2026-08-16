"""Module for parsing drug and ingredient data from FAERS records.

This module provides functionality to extract and parse drug and ingredient
information from dataframes, including named entity recognition using NLP models.
"""

import logging
from typing import Any
from pyspark.sql import DataFrame
import pyspark.sql.functions as F
from tgedr_dataops_abs.processor import Processor
from pvprototypes_faers.nlp.entity_parsing import parse_entities_udf

logger = logging.getLogger(__name__)


class DrugIngredientEntity(Processor):
    """Extract named entities from text using an NLP model.

    This class extends the Processor class and provides functionality to
    parse drug and ingredient data from FAERS records using named entity
    recognition (NER) techniques.

    it returns a dataframe with original columns plus a new column 'entity'
    which contains the extracted entities from the 'term' column which is derived from the fields "ingredient" and "drug".
    [ drugname, prod_ai, term, entity ]
    """

    CONTEXT_KEY_DF = "dataframe"

    def process(self, context: dict[str, Any] | None = None) -> DataFrame:
        """Process drug and ingredient data by parsing named entities.

        Args:
            context: Dictionary containing 'dataframe' key with input DataFrame.

        Returns:
            DataFrame with original columns plus 'entity' column.

        Raises:
            Exception: If context is missing or does not contain 'dataframe' key.
        """
        logger.info(f"[process|in] ({context})")
        if not context or self.CONTEXT_KEY_DF not in context:
            raise Exception(f"{self.CONTEXT_KEY_DF} must be provided in context")
        df: DataFrame = context[self.CONTEXT_KEY_DF]

        df_di = df.select(
            F.lower("drugname").alias("drugname"),
            F.lower("prod_ai").alias("prod_ai"),
            F.lower(
                F.regexp_replace("drugname", r"([\(\)\/\+])", " $1 "),
            ).alias("drug"),
            F.lower(
                F.regexp_replace("prod_ai", r"([\(\)\/\+])", " $1 "),
            ).alias("ingredient"),
        ).distinct()

        # let's assume if there is ingredient then we can discard drugname
        df_term = df_di.select(
            "drugname",
            "prod_ai",
            F.when(F.col("ingredient").isNotNull(), F.col("ingredient")).otherwise(F.col("drug")).alias("term"),
        ).distinct()

        result: DataFrame = (
            df_term.withColumn("parsed_entities", parse_entities_udf(F.col("term")))
            .withColumn("entity", F.explode_outer("parsed_entities"))
            .drop("parsed_entities")
        )

        logger.info(f"[process|out] => {result}")
        return result

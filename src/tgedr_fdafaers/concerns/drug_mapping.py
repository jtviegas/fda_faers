"""Enrich bronze FAERS datasets with ATC mappings and save them to silver."""

import logging
from typing import Any
from pyspark.sql import DataFrame

from tgedr_dataops_abs.processor import Processor


logger = logging.getLogger(__name__)


class DrugMapping(Processor):
    """Enrich bronze FAERS datasets with ATC mappings and save them to silver."""

    CONTEXT_KEY_TERM_ATC_DF = "term_atc_dataset"
    CONTEXT_KEY_DRUG_DF = "drug_dataset"

    def process(self, context: dict[str, Any] | None = None) -> DataFrame | None:
        """Enrich the drug dataset with ATC mappings by joining on the term column.

        Args:
            context: The processing context containing the term-ATC and drug datasets.

        Returns:
            The drug dataset left-joined with the term-ATC dataset, or None.

        Raises:
            ValueError: If a required context key is missing.
        """
        logger.info(f"[process|in] ({context})")
        result: DataFrame | None = None

        if self.CONTEXT_KEY_TERM_ATC_DF not in context:
            raise ValueError(f"[process] Missing required context key: {self.CONTEXT_KEY_TERM_ATC_DF}")
        df_term_atc: DataFrame = context[self.CONTEXT_KEY_TERM_ATC_DF]

        if self.CONTEXT_KEY_DRUG_DF not in context:
            raise ValueError(f"[process] Missing required context key: {self.CONTEXT_KEY_DRUG_DF}")
        df_drug: DataFrame = context[self.CONTEXT_KEY_DRUG_DF]

        result = df_drug.join(df_term_atc, on="term", how="left")

        logger.info(f"[process|out] => {result}")
        return result

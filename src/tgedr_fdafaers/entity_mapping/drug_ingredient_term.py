"""Processor that builds a deduplicated term mapping from drug names and product active ingredients."""

import logging
from typing import Any

import pandas as pd

from tgedr_dataops_abs.processor import Processor, ProcessorException

logger = logging.getLogger(__name__)


class DrugIngredientTerm(Processor):
    """Processor that builds a deduplicated term mapping from drug names and product active ingredients."""

    CONTEXT_KEY_DF = "dataframe"
    CONTEXT_KEY_TERM_COLUMN = "term_column"

    def process(self, context: dict[str, Any] | None = None) -> pd.DataFrame:
        """Build a deduplicated term mapping from drug names and product active ingredients.
        The resulting dataframe will have an additional column named ``term`` containing
        the normalized ingredient when available, otherwise the normalized drug name.

        Args:
            context: Processing context that must contain a dataframe under the
                ``CONTEXT_KEY_DF`` key, with ``drugname`` and ``prod_ai`` columns. Optionally, the context can also contain
                ``CONTEXT_KEY_TERM_COLUMN`` key specifying the name of the term column in the resulting dataframe.

        Returns:
            A dataframe with an additional column named ``term`` (or the name specified
            in ``CONTEXT_KEY_TERM_COLUMN``) containing the normalized ingredient when
            available, otherwise the normalized drug name.

        Raises:
            ProcessorException: If ``context`` is missing or does not contain the
                required dataframe.
        """
        logger.info(f"[process|in] ({context})")
        if not context or self.CONTEXT_KEY_DF not in context:
            raise ProcessorException(f"{self.CONTEXT_KEY_DF} must be provided in context")
        df: pd.DataFrame = context[self.CONTEXT_KEY_DF]

        df_di = df.assign(
            drug=df["drugname"]
            .str.lower()
            .str.replace(r"([\(\)\/\+])", r" \1 ", regex=True)
            .str.replace(r"/", "", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip(),
            ingredient=df["prod_ai"]
            .str.lower()
            .str.replace(r"([\(\)\/\+])", r" \1 ", regex=True)
            .str.replace(r"/", "", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip(),
        )

        # let's assume if there is ingredient then we can discard drugname
        df_term = (
            df_di.assign(term=df_di["ingredient"].where(df_di["ingredient"].notna(), df_di["drug"]))
            .drop(columns=["drug", "ingredient"])
            .drop_duplicates()
        )

        if self.CONTEXT_KEY_TERM_COLUMN in context:
            df_term = df_term.rename(columns={"term": context[self.CONTEXT_KEY_TERM_COLUMN]})

        logger.info(f"[process|out] => {df_term.shape}")
        return df_term

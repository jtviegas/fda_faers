"""Module providing AiAtcMapping — a DataFrame-level ATC mapping backed by Databricks ai_query.

The implementation avoids redundant LLM calls by deduplicating terms before
sending them to the model, then joining the results back to the original DataFrame.
"""

import logging

import pyspark.sql.functions as F
from pyspark.sql import DataFrame

from pvprototypes_faers.data_mapping.atc_mapping.mapping import AtcMapping


logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "databricks-meta-llama-3-3-70b-instruct"

_PROMPT_PREFIX = (
    "You are a seasoned pharmacovigilance expert with deep knowledge of drug "
    "classifications and the ATC (Anatomical Therapeutic Chemical) system. "
    "You are provided with a drug or ingredient name extracted from a FAERS record. "
    "Your task is to determine the 6 ATC codes most closely related to the ingredient. "
    "Return only the ATC codes as a formatted json string with "
    "a list of ATC codes defined by its attributes `id`, `name`, and `class_type` as in: "
    "[{'id': 'C01EB', 'name': 'Other cardiac preparations', 'class_type': 'ATC1-4'}, {'id': 'A11HA', 'name': 'Other plain vitamin preparations', 'class_type': 'ATC1-4'}]"
    "Do not include commentary or suggestions. "
    "If you cannot determine any relevant ATC codes, return null.\n\n"
    "Drug or ingredient: "
)


class AiAtcMapping(AtcMapping):
    """Decorates a DataFrame with ATC codes using the Databricks ``ai_query`` function.

    Unique terms are resolved first (deduplication) so that each distinct drug
    name is sent to the LLM exactly once. The mapped results are then joined
    back to the original DataFrame, preserving all rows and columns.

    Example::

        mapping = AiAtcMapping(term_col="drug_name", output_col="atc_codes")
        decorated_df = mapping.decorate(drug_df)
    """

    def __init__(
        self,
        term_col: str = "term",
        output_col: str = "atc_codes",
        model: str = _DEFAULT_MODEL,
    ) -> None:
        """Initialise with column names and the Databricks Foundation Model endpoint.

        Args:
            term_col: DataFrame column holding drug/ingredient term strings.
            output_col: Name of the column added with the AI-resolved ATC codes.
            model: Databricks Foundation Model endpoint name passed to ``ai_query``.
        """
        super().__init__(term_col=term_col, output_col=output_col)
        self._model = model

    def _ai_column(self, df: DataFrame) -> DataFrame:
        """Apply ``ai_query`` to the ``_prompt`` column and store the result.

        Separated into its own method so that subclasses (and tests) can override
        the AI call without touching the deduplication / join logic.

        Args:
            df: DataFrame with a ``_prompt`` column containing the full prompt string.

        Returns:
            The same DataFrame with ``self._output_col`` added.
        """
        return df.withColumn(
            self._output_col,
            F.expr(f"ai_query('{self._model}', _prompt)"),
        )

    def decorate(self, df: DataFrame) -> DataFrame:
        """Decorate *df* with ATC codes resolved via Databricks ``ai_query``.

        Steps:
        1. Select distinct values of ``term_col`` to avoid duplicate LLM calls.
        2. Build a prompt string for each unique term.
        3. Call ``ai_query`` once per unique term.
        4. Left-join the results back to the original DataFrame.

        Args:
            df: Input DataFrame. Must contain a string column named ``self._term_col``.

        Returns:
            Original DataFrame with ``self._output_col`` appended.
        """
        logger.info(
            f"[decorate|in] term_col={self._term_col!r}, output_col={self._output_col!r}, model={self._model!r}"
        )

        unique_terms: DataFrame = df.select(self._term_col).distinct()

        prompted = unique_terms.withColumn(
            "_prompt",
            F.concat(F.lit(_PROMPT_PREFIX), F.col(self._term_col)),
        )

        mapped = (
            self._ai_column(prompted)
            .drop("_prompt")
            .withColumn(self._output_col, F.from_json(F.col(self._output_col), AtcMapping.ATC_CODE_SCHEMA))
            .withColumn("strategy", F.lit(self.__class__.__name__))
        )

        result = df.join(mapped, on=self._term_col, how="left")

        logger.info(f"[decorate|out] => added column {self._output_col!r}")
        return result

"""
Module for mapping MedDRA preferred terms and codes to FAERS datasets.

This module provides the MedDRAMapping processor, which maps MedDRA preferred terms
and codes to FAERS reaction and indication datasets for pharmacovigilance analysis.
"""

from types import MappingProxyType
from typing import Any, Final
import logging
from pyspark.sql import DataFrame
import pyspark.sql.functions as F
from tgedr_dataops_abs.processor import Processor


logger = logging.getLogger(__name__)


class MedDRAMapping(Processor):
    """
    Processor for mapping MedDRA preferred terms and codes to FAERS datasets.

    This class processes FAERS and MedDRA datasets by mapping preferred terms and codes, ensuring that the resulting FAERS datasets contain the appropriate MedDRA information for further analysis.
    """

    __FAERS_DATASETS: tuple[str, ...] = ("reac", "indi")
    __MEDDRA_DATASETS: tuple[str, ...] = ("llt_former_pt", "preferred_terms")  # , "pt_nnmq", "pt_smq"
    __COLS: MappingProxyType[str, tuple[str, ...]] = MappingProxyType(
        {
            "reac": (
                "caseid",
                "primaryid",
                "drug_rec_act",
                "pt",
                "processing_time",
                "period",
                "pt_code",
                "pt_soc_code",
                "nnmq_codes",
                "smq_codes",
            ),
            "indi": (
                "caseid",
                "primaryid",
                "indi_drug_seq",
                "indi_pt",
                "processing_time",
                "period",
                "indi_pt_code",
                "indi_pt_soc_code",
                "nnmq_codes",
                "smq_codes",
            ),
        }
    )

    CONTEXT_KEY_DATASET: Final[str] = "dataset"
    CONTEXT_KEY_TABLE: Final[str] = "table"
    CONTEXT_KEY_MEDDRA_DATASETS: Final[str] = "meddra_datasets"

    def process(self, context: dict[str, Any] | None = None) -> DataFrame:
        """
        Map MedDRA preferred terms and codes to a FAERS dataset.

        Args:
            context: A dictionary containing:
                - ``dataset``: The input FAERS DataFrame to process.
                - ``table``: The name of the FAERS table (``"reac"`` or ``"indi"``).
                - ``meddra_datasets``: A dictionary of MedDRA DataFrames keyed by
                  ``"llt_former_pt"``, ``"preferred_terms"``, ``"pt_nnmq"``, and ``"pt_smq"``.

        Returns:
            A DataFrame with MedDRA preferred term codes and SOC codes mapped to the input dataset.

        Raises:
            Exception: If required context keys are missing or invalid.
        """
        logger.info(f"[process|in] ({context})")

        if not context or self.CONTEXT_KEY_DATASET not in context:
            raise Exception(f"[process] configuration must include '{self.CONTEXT_KEY_DATASET}'")
        if self.CONTEXT_KEY_TABLE not in context:
            raise Exception(f"[process] configuration must include '{self.CONTEXT_KEY_TABLE}'")
        if self.CONTEXT_KEY_MEDDRA_DATASETS not in context:
            raise Exception(f"[process] configuration must include '{self.CONTEXT_KEY_MEDDRA_DATASETS}'")
        if sorted(self.__MEDDRA_DATASETS) != sorted((context[self.CONTEXT_KEY_MEDDRA_DATASETS]).keys()):
            raise Exception(f"[process] configuration MedDRA datasets must have keys: {self.__MEDDRA_DATASETS}")

        df_table: DataFrame = context[self.CONTEXT_KEY_DATASET]
        table: str = context[self.CONTEXT_KEY_TABLE]
        if table not in self.__FAERS_DATASETS:
            raise Exception(f"[process] not a known fears table: {table}")
        meddra_datasets: dict[str, DataFrame] = context[self.CONTEXT_KEY_MEDDRA_DATASETS]

        result: DataFrame = self._match_preferred_terms(
            df=df_table,
            df_pt=meddra_datasets["preferred_terms"],
            df_llt_former_pt=meddra_datasets["llt_former_pt"],
            output_cols=list(self.__COLS[table]),
            pt_col="indi_pt" if table == "indi" else "pt",
            pt_code_col="indi_pt_code" if table == "indi" else "pt_code",
            pt_soc_code_col="indi_pt_soc_code" if table == "indi" else "pt_soc_code",
        )
        logger.info(f"[process|out] => {result}")
        return result

    def _match_preferred_terms(
        self,
        df: DataFrame,
        df_pt: DataFrame,
        df_llt_former_pt: DataFrame,
        output_cols: list[str],
        pt_col: str = "pt",
        pt_code_col: str = "pt_code",
        pt_soc_code_col: str = "pt_soc_code",
    ) -> DataFrame:
        logger.info(
            f"[_match_preferred_terms|in] ({df}, {df_pt}, {df_llt_former_pt}, {output_cols}, {pt_col}, {pt_code_col}, {pt_soc_code_col})"
        )

        cols_to_do = ["pt_code", pt_code_col, "pt_soc_code", pt_soc_code_col, "nnmq_codes", "smq_codes"]
        # match with the pt mapping dataset
        df_pt_matches = (
            df.drop(*cols_to_do)
            .withColumnRenamed(pt_col, "pt")
            .withColumn("pt", F.lower(F.col("pt")))
            .join(df_pt.drop("processing_time"), on="pt", how="left_outer")
        )
        # separate the unmatched data
        df_unmatched = df_pt_matches.filter(F.col("pt_code").isNull()).drop(*cols_to_do)
        df_matched = (
            df_pt_matches.filter(F.col("pt_code").isNotNull())
            .withColumnRenamed("pt", pt_col)
            .withColumnRenamed("pt_code", pt_code_col)
            .withColumnRenamed("pt_soc_code", pt_soc_code_col)
            .select(*output_cols)
        )

        # try to fill the unmatched with llt's that were pt's
        df_unmatched = (
            df_unmatched.join(df_llt_former_pt, df_unmatched["pt"] == df_llt_former_pt["llt_name"], "left_outer")
            .drop("llt_name", "llt_code")
            .withColumnRenamed("pt", pt_col)
            .withColumnRenamed("pt_code", pt_code_col)
            .withColumnRenamed("pt_soc_code", pt_soc_code_col)
        ).select(*output_cols)

        # join the data back together
        result: DataFrame = df_matched.union(df_unmatched)

        logger.info(f"[_match_preferred_terms|out] => {result}")
        return result

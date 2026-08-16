"""
This module provides the RawDataIngestion processor for ingesting and transforming raw FAERS input data into a standardized format using PySpark.
"""

import logging
from types import MappingProxyType
from typing import Any
from pyspark.sql.types import StructType, StructField, StringType, LongType, DoubleType
from pyspark.sql import DataFrame, functions as F
from pyspark.sql.types import ArrayType
from tgedr_dataops_abs.processor import Processor

logger = logging.getLogger(__name__)


class RawDataIngestion(Processor):
    """Processor for ingesting and transforming raw FAERS input data into a standardized format."""

    CONTEXT_KEY_TABLE = "table"
    CONTEXT_KEY_DATAFRAME = "dataframe"

    __SCHEMAS = MappingProxyType(
        {
            "demo": StructType(
                [
                    StructField("primaryid", LongType(), nullable=True),
                    StructField("caseid", LongType(), nullable=True),
                    StructField("caseversion", LongType(), nullable=True),
                    StructField("i_f_cod", StringType(), nullable=True),
                    StructField("event_dt", LongType(), nullable=True),
                    StructField("mfr_dt", LongType(), nullable=True),
                    StructField("init_fda_dt", LongType(), nullable=True),
                    StructField("fda_dt", LongType(), nullable=True),
                    StructField("rept_cod", StringType(), nullable=True),
                    StructField("auth_num", StringType(), nullable=True),
                    StructField("mfr_num", StringType(), nullable=True),
                    StructField("mfr_sndr", StringType(), nullable=True),
                    StructField("lit_ref", StringType(), nullable=True),
                    StructField("age", LongType(), nullable=True),
                    StructField("age_cod", StringType(), nullable=True),
                    StructField("age_grp", StringType(), nullable=True),
                    StructField("sex", StringType(), nullable=True),
                    StructField("e_sub", StringType(), nullable=True),
                    StructField("wt", DoubleType(), nullable=True),
                    StructField("wt_cod", StringType(), nullable=True),
                    StructField("rept_dt", LongType(), nullable=True),
                    StructField("to_mfr", StringType(), nullable=True),
                    StructField("occp_cod", StringType(), nullable=True),
                    StructField("reporter_country", StringType(), nullable=True),
                    StructField("occr_country", StringType(), nullable=True),
                    StructField("processing_time", LongType(), nullable=True),
                    StructField("period", StringType(), nullable=True),
                ]
            ),
            "drug": StructType(
                [
                    StructField("primaryid", LongType(), nullable=True),
                    StructField("caseid", LongType(), nullable=True),
                    StructField("drug_seq", LongType(), nullable=True),
                    StructField("role_cod", StringType(), nullable=True),
                    StructField("drugname", StringType(), nullable=True),
                    StructField("prod_ai", StringType(), nullable=True),
                    StructField("val_vbm", LongType(), nullable=True),
                    StructField("route", StringType(), nullable=True),
                    StructField("dose_vbm", StringType(), nullable=True),
                    StructField("cum_dose_chr", StringType(), nullable=True),
                    StructField("cum_dose_unit", StringType(), nullable=True),
                    StructField("dechal", StringType(), nullable=True),
                    StructField("rechal", StringType(), nullable=True),
                    StructField("lot_num", StringType(), nullable=True),
                    StructField("exp_dt", LongType(), nullable=True),
                    StructField("nda_num", DoubleType(), nullable=True),
                    StructField("dose_amt", StringType(), nullable=True),
                    StructField("dose_unit", StringType(), nullable=True),
                    StructField("dose_form", StringType(), nullable=True),
                    StructField("dose_freq", StringType(), nullable=True),
                    StructField("processing_time", LongType(), nullable=True),
                    StructField("period", StringType(), nullable=True),
                ]
            ),
            "reac": StructType(
                [
                    StructField("primaryid", LongType(), nullable=True),
                    StructField("caseid", LongType(), nullable=True),
                    StructField("pt", StringType(), nullable=True),
                    StructField("drug_rec_act", StringType(), nullable=True),
                    StructField("processing_time", LongType(), nullable=True),
                    StructField("period", StringType(), nullable=True),
                ]
            ),
            "outc": StructType(
                [
                    StructField("primaryid", LongType(), nullable=True),
                    StructField("caseid", LongType(), nullable=True),
                    StructField("outc_cod", StringType(), nullable=True),
                    StructField("processing_time", LongType(), nullable=True),
                    StructField("period", StringType(), nullable=True),
                ]
            ),
            "rpsr": StructType(
                [
                    StructField("primaryid", LongType(), nullable=True),
                    StructField("caseid", LongType(), nullable=True),
                    StructField("rpsr_cod", StringType(), nullable=True),
                    StructField("processing_time", LongType(), nullable=True),
                    StructField("period", StringType(), nullable=True),
                ]
            ),
            "ther": StructType(
                [
                    StructField("primaryid", LongType(), nullable=True),
                    StructField("caseid", LongType(), nullable=True),
                    StructField("dsg_drug_seq", LongType(), nullable=True),
                    StructField("start_dt", LongType(), nullable=True),
                    StructField("end_dt", LongType(), nullable=True),
                    StructField("dur", DoubleType(), nullable=True),
                    StructField("dur_cod", StringType(), nullable=True),
                    StructField("processing_time", LongType(), nullable=True),
                    StructField("period", StringType(), nullable=True),
                ]
            ),
            "indi": StructType(
                [
                    StructField("primaryid", LongType(), nullable=True),
                    StructField("caseid", LongType(), nullable=True),
                    StructField("indi_drug_seq", LongType(), nullable=True),
                    StructField("indi_pt", StringType(), nullable=True),
                    StructField("processing_time", LongType(), nullable=True),
                    StructField("period", StringType(), nullable=True),
                ]
            ),
        }
    )

    __COLUMN_CASTS = {
        "age": LongType(),
        "age_cod": StringType(),
        "age_grp": StringType(),
        "auth_num": StringType(),
        "caseid": LongType(),
        "caseversion": LongType(),
        "concept_ids": ArrayType(StringType()),
        "concept_class_ids": ArrayType(StringType()),
        "concept_name": StringType(),
        "cum_dose_unit": StringType(),
        "cum_dose_chr": StringType(),
        "dechal": StringType(),
        "dose_amt": StringType(),
        "dose_form": StringType(),
        "dose_freq": StringType(),
        "dose_unit": StringType(),
        "dose_vbm": StringType(),
        "drug_rec_act": StringType(),
        "drug_seq": LongType(),
        "drugname": StringType(),
        "dsg_drug_seq": LongType(),
        "dur": DoubleType(),
        "dur_cod": StringType(),
        "e_sub": StringType(),
        "end_dt": LongType(),
        "event_dt": LongType(),
        "exp_dt": LongType(),
        "fda_dt": LongType(),
        "i_f_cod": StringType(),
        "indi_drug_seq": LongType(),
        "indi_pt": StringType(),
        "indi_pt_code": LongType(),
        "indi_pt_soc_code": LongType(),
        "init_fda_dt": LongType(),
        "lit_ref": StringType(),
        "lot_num": StringType(),
        "mfr_dt": LongType(),
        "mfr_num": StringType(),
        "mfr_sndr": StringType(),
        "nda_num": DoubleType(),
        "nnmq_codes": ArrayType(LongType()),
        "occp_cod": StringType(),
        "occr_country": StringType(),
        "outc_cod": StringType(),
        "period": StringType(),
        "primaryid": LongType(),
        "processing_time": LongType(),
        "prod_ai": StringType(),
        "pt": StringType(),
        "pt_code": LongType(),
        "pt_soc_code": LongType(),
        "rechal": StringType(),
        "reporter_country": StringType(),
        "rept_cod": StringType(),
        "rept_dt": LongType(),
        "role_cod": StringType(),
        "route": StringType(),
        "rpsr_cod": StringType(),
        "sex": StringType(),
        "smq_codes": ArrayType(LongType()),
        "start_dt": LongType(),
        "to_mfr": StringType(),
        "val_vbm": LongType(),
        "wt": DoubleType(),
        "wt_cod": StringType(),
    }

    def get_schema(self, table: str) -> StructType:
        """
        Retrieve the predefined schema for the specified table.

        Args:
            table (str): The name of the table for which to get the schema.

        Returns:
            StructType: The schema associated with the table.

        Raises:
            ValueError: If no predefined schema exists for the given table.
        """
        if table not in self.__SCHEMAS:
            raise ValueError(f"No predefined schema for table '{table}'")
        return self.__SCHEMAS[table]

    def _validate_schema(self, df: DataFrame, table: str):
        """Validate schema before processing."""
        logger.info(f"[_validate_schema|in] (df={df}, table={table})")
        expected = self.__SCHEMAS[table]
        if df.schema != expected:
            raise ValueError(
                f"Schema mismatch for {table}: expected {[f.name for f in expected.fields]}, got {df.columns}"
            )
        logger.info("[_validate_schema|out]")

    def _apply_column_casts(self, df: DataFrame) -> DataFrame:
        # types are accordingly to documentation in fda faers files
        logger.info(f"[_apply_column_casts|in] (df={df})")
        for column, _type in self.__COLUMN_CASTS.items():
            if column in df.columns:
                if _type.__class__.__name__ == "LongType":
                    df = df.withColumn(column, F.col(column).cast(DoubleType()).cast(_type))
                else:
                    df = df.withColumn(column, F.col(column).cast(_type))
        logger.info(f"[_apply_column_casts|out] => {df.columns}")
        return df

    def _handle_nulls(self, df: DataFrame) -> DataFrame:
        """Handle null values by filling them with default values based on column type."""
        logger.info(f"[_handle_nulls|in] (df={df})")

        string_cols = [f.name for f in df.schema.fields if isinstance(f.dataType, StringType)]
        result: DataFrame = df.select(
            *[
                F.when(F.col(c).isin(["None", "nan"]), None).otherwise(F.col(c)).alias(c)
                if c in string_cols
                else F.col(c)
                for c in df.columns
            ]
        )
        result = result.replace(float("nan"), None)

        logger.info(f"[_handle_nulls|out] => {result.columns}")
        return result

    def _sort_columns(self, df: DataFrame, table: str) -> DataFrame:
        """Sort columns according to predefined schema order."""
        logger.info(f"[_sort_columns|in] (df={df}, table={table})")
        if table in self.__SCHEMAS:
            schema = self.__SCHEMAS[table]
            column_order = [field.name for field in schema.fields if field.name in df.columns]
            column_order += [col for col in df.columns if col not in column_order]  # add any extra columns at the end
            df = df.select(column_order)
            logger.info(f"[_sort_columns|out] => {df.columns}")
        else:
            logger.warning(f"No predefined schema for table '{table}'. Columns will not be sorted.")
        return df

    def process(self, context: dict[str, Any] | None = None) -> DataFrame:
        """
        Process the input DataFrame according to the table-specific schema and transformations.
        Requires 'table' and 'dataframe' keys in the context dictionary.
        """
        logger.info(f"[process|in] ({context})")

        if (
            not context
            or (self.CONTEXT_KEY_TABLE not in context.keys())
            or (self.CONTEXT_KEY_DATAFRAME not in context.keys())
        ):
            raise Exception("you must provide 'table' and 'dataframe' in context")

        table: str = context[self.CONTEXT_KEY_TABLE]
        df = context[self.CONTEXT_KEY_DATAFRAME]
        df = self._apply_column_casts(df).drop("_rescued_data")
        df = self._handle_nulls(df)
        df = self._sort_columns(df, table)
        self._validate_schema(df, table)

        logger.info(f"[process|out] => {df}")
        return df

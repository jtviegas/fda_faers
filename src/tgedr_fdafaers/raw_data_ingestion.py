"""Ingest and transform raw FAERS data into standardized dataframes."""

import logging
from types import MappingProxyType
from typing import Any, ClassVar

import numpy as np
import pandas as pd
from tgedr_dataops_abs.processor import Processor

logger = logging.getLogger(__name__)


class RawDataIngestion(Processor):
    """Processor for ingesting and transforming raw FAERS input data into a standardized format."""

    CONTEXT_KEY_TABLE = "table"
    CONTEXT_KEY_DATAFRAME = "dataframe"

    __SCHEMAS = MappingProxyType(
        {
            "demo": {
                "primaryid": "Int64",
                "caseid": "Int64",
                "caseversion": "Int64",
                "i_f_cod": "string",
                "event_dt": "Int64",
                "mfr_dt": "Int64",
                "init_fda_dt": "Int64",
                "fda_dt": "Int64",
                "rept_cod": "string",
                "auth_num": "string",
                "mfr_num": "string",
                "mfr_sndr": "string",
                "lit_ref": "string",
                "age": "Int64",
                "age_cod": "string",
                "age_grp": "string",
                "sex": "string",
                "e_sub": "string",
                "wt": "Float64",
                "wt_cod": "string",
                "rept_dt": "Int64",
                "to_mfr": "string",
                "occp_cod": "string",
                "reporter_country": "string",
                "occr_country": "string",
                "processing_time": "Int64",
                "period": "string",
            },
            "drug": {
                "primaryid": "Int64",
                "caseid": "Int64",
                "drug_seq": "Int64",
                "role_cod": "string",
                "drugname": "string",
                "prod_ai": "string",
                "val_vbm": "Int64",
                "route": "string",
                "dose_vbm": "string",
                "cum_dose_chr": "string",
                "cum_dose_unit": "string",
                "dechal": "string",
                "rechal": "string",
                "lot_num": "string",
                "exp_dt": "Int64",
                "nda_num": "Float64",
                "dose_amt": "string",
                "dose_unit": "string",
                "dose_form": "string",
                "dose_freq": "string",
                "processing_time": "Int64",
                "period": "string",
            },
            "reac": {
                "primaryid": "Int64",
                "caseid": "Int64",
                "pt": "string",
                "drug_rec_act": "string",
                "processing_time": "Int64",
                "period": "string",
            },
            "outc": {
                "primaryid": "Int64",
                "caseid": "Int64",
                "outc_cod": "string",
                "processing_time": "Int64",
                "period": "string",
            },
            "rpsr": {
                "primaryid": "Int64",
                "caseid": "Int64",
                "rpsr_cod": "string",
                "processing_time": "Int64",
                "period": "string",
            },
            "ther": {
                "primaryid": "Int64",
                "caseid": "Int64",
                "dsg_drug_seq": "Int64",
                "start_dt": "Int64",
                "end_dt": "Int64",
                "dur": "Float64",
                "dur_cod": "string",
                "processing_time": "Int64",
                "period": "string",
            },
            "indi": {
                "primaryid": "Int64",
                "caseid": "Int64",
                "indi_drug_seq": "Int64",
                "indi_pt": "string",
                "processing_time": "Int64",
                "period": "string",
            },
        }
    )

    __COLUMN_CASTS: ClassVar[dict[str, str]] = {
        "age": "Int64",
        "age_cod": "string",
        "age_grp": "string",
        "auth_num": "string",
        "caseid": "Int64",
        "caseversion": "Int64",
        "cum_dose_unit": "string",
        "cum_dose_chr": "string",
        "dechal": "string",
        "dose_amt": "string",
        "dose_form": "string",
        "dose_freq": "string",
        "dose_unit": "string",
        "dose_vbm": "string",
        "drug_rec_act": "string",
        "drug_seq": "Int64",
        "drugname": "string",
        "dsg_drug_seq": "Int64",
        "dur": "Float64",
        "dur_cod": "string",
        "e_sub": "string",
        "end_dt": "Int64",
        "event_dt": "Int64",
        "exp_dt": "Int64",
        "fda_dt": "Int64",
        "i_f_cod": "string",
        "indi_drug_seq": "Int64",
        "indi_pt": "string",
        "init_fda_dt": "Int64",
        "lit_ref": "string",
        "lot_num": "string",
        "mfr_dt": "Int64",
        "mfr_num": "string",
        "mfr_sndr": "string",
        "nda_num": "Float64",
        "occp_cod": "string",
        "occr_country": "string",
        "outc_cod": "string",
        "period": "string",
        "primaryid": "Int64",
        "processing_time": "Int64",
        "prod_ai": "string",
        "pt": "string",
        "rechal": "string",
        "reporter_country": "string",
        "rept_cod": "string",
        "rept_dt": "Int64",
        "role_cod": "string",
        "route": "string",
        "rpsr_cod": "string",
        "sex": "string",
        "start_dt": "Int64",
        "to_mfr": "string",
        "val_vbm": "Int64",
        "wt": "Float64",
        "wt_cod": "string",
    }

    def get_schema(self, table: str) -> dict[str, str]:
        """
        Retrieve the predefined schema for the specified table.

        Args:
            table (str): The name of the table for which to get the schema.

        Returns:
            dict[str, str]: The schema (column name -> dtype) associated with the table.

        Raises:
            ValueError: If no predefined schema exists for the given table.
        """
        if table not in self.__SCHEMAS:
            raise ValueError(f"No predefined schema for table '{table}'")
        return dict(self.__SCHEMAS[table])

    def _validate_schema(self, df: pd.DataFrame, table: str) -> None:
        """Validate schema before processing."""
        logger.info(f"[_validate_schema|in] (table={table})")
        expected = self.__SCHEMAS[table]
        expected_cols = list(expected.keys())
        if list(df.columns) != expected_cols:
            raise ValueError(
                f"Schema mismatch for {table}: expected {expected_cols}, got {list(df.columns)}"
            )
        logger.info("[_validate_schema|out]")

    def _apply_column_casts(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cast columns to the types defined in __COLUMN_CASTS."""
        logger.info(f"[_apply_column_casts|in] (columns={list(df.columns)})")
        for column, dtype in self.__COLUMN_CASTS.items():
            if column in df.columns:
                if dtype == "Int64":
                    # Convert via float first to handle string representations
                    df[column] = pd.to_numeric(df[column], errors="coerce").astype("Int64")
                elif dtype == "Float64":
                    df[column] = pd.to_numeric(df[column], errors="coerce").astype("Float64")
                else:
                    df[column] = df[column].astype("string")
        logger.info(f"[_apply_column_casts|out] => {list(df.columns)}")
        return df

    def _handle_nulls(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle null values by replacing sentinel strings and NaN with pd.NA."""
        logger.info(f"[_handle_nulls|in] (columns={list(df.columns)})")

        string_cols = df.select_dtypes(include=["string"]).columns.tolist()
        for col in string_cols:
            df[col] = df[col].replace(["None", "nan"], pd.NA)

        # Replace NaN in numeric columns with pd.NA
        df = df.replace({np.nan: pd.NA})

        logger.info(f"[_handle_nulls|out] => {list(df.columns)}")
        return df

    def _sort_columns(self, df: pd.DataFrame, table: str) -> pd.DataFrame:
        """Sort columns according to predefined schema order."""
        logger.info(f"[_sort_columns|in] (table={table})")
        if table in self.__SCHEMAS:
            schema = self.__SCHEMAS[table]
            column_order = [col for col in schema if col in df.columns]
            column_order += [col for col in df.columns if col not in column_order]
            df = df[column_order]
            logger.info(f"[_sort_columns|out] => {list(df.columns)}")
        else:
            logger.warning(f"No predefined schema for table '{table}'. Columns will not be sorted.")
        return df

    def process(self, context: dict[str, Any] | None = None) -> pd.DataFrame:
        """
        Process the input DataFrame according to the table-specific schema and transformations.
        Requires 'table' and 'dataframe' keys in the context dictionary.
        """
        logger.info(f"[process|in] ({context})")

        if (
            not context
            or (self.CONTEXT_KEY_TABLE not in context)
            or (self.CONTEXT_KEY_DATAFRAME not in context)
        ):
            msg = "you must provide 'table' and 'dataframe' in context"
            raise Exception(msg)  # noqa: TRY002

        table: str = context[self.CONTEXT_KEY_TABLE]
        df: pd.DataFrame = context[self.CONTEXT_KEY_DATAFRAME]
        if "_rescued_data" in df.columns:
            df = df.drop(columns=["_rescued_data"])
        df = self._apply_column_casts(df)
        df = self._handle_nulls(df)
        df = self._sort_columns(df, table)
        self._validate_schema(df, table)

        logger.info(f"[process|out] => {list(df.columns)}")
        return df

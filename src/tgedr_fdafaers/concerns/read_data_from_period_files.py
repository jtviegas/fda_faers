"""Process and load corrected FAERS period files."""

from typing import Any
import pandas as pd
from pathlib import Path
import logging

from tgedr_dataops_abs.processor import Processor
from tgedr_fdafaers.raw_data_ingestion import RawDataIngestion
from tgedr_fdafaers.entity_mapping.drug_ingredient_term import DrugIngredientTerm


logger = logging.getLogger(__name__)


class ReadDataFromPeriodFiles(Processor):
    """ETL workflow for extracting, correcting, and loading FAERS period files."""

    CONTEXT_KEY_FILES = "files"

    def process(self, context: dict[str, Any] | None = None) -> dict[str, pd.DataFrame] | None:
        """Process FAERS period files contained in the context.

        Args:
            context: Dictionary containing required context keys. Must include
                the 'files' key with a list of file paths to ingest.

        Returns:
            A dictionary mapping table names to their resulting DataFrames, or
            None if no files were provided.
        """
        logger.info(f"[process|in] ({context})")

        result: dict[str, pd.DataFrame] | None = None

        if self.CONTEXT_KEY_FILES not in context:
            raise ValueError(f"[process] Missing required context key: {self.CONTEXT_KEY_FILES}")
        files: list[str] = context[self.CONTEXT_KEY_FILES]

        if 0 < len(files):
            result = {}
            for file in files:
                self.__handle_file(file.strip(), result)
            drug_ingredient_term: DrugIngredientTerm = DrugIngredientTerm()
            result["drug"] = drug_ingredient_term.process(context={"dataframe": result["drug"]})

        logger.info(f"[process|out] => {result}")
        return result

    def __handle_file(self, filepath: str, data: dict[str, pd.DataFrame]) -> None:
        """Handle the ingestion of a single FAERS text file."""
        logger.info(f"[__handle_file|in] ({filepath}, {data})")

        file = Path(filepath).name
        table = file[0:4]
        df: pd.DataFrame = pd.read_csv(
            filepath, delimiter=self._constants.CSV_DELIMITER, index_col=False, low_memory=False
        )
        result: pd.DataFrame = RawDataIngestion().process(
            context={RawDataIngestion.CONTEXT_KEY_TABLE: table, RawDataIngestion.CONTEXT_KEY_DATAFRAME: df}
        )

        if table not in data:
            data[table] = result
        else:
            data[table] = pd.concat([data[table], result], ignore_index=True)
        logger.info(f"[__handle_file|out] new data | table: {table} | df.shape: {result.shape}")

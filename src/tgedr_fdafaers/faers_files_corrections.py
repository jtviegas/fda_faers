"""Processor that applies corrections to raw FAERS files before ingestion."""

from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any, ClassVar
import pandas as pd
import logging

from tgedr_dataops_abs.processor import Processor

from tgedr_fdafaers.constants import Constants
from tgedr_fdafaers.utils.faers_period import FaersPeriod


logger = logging.getLogger(__name__)


class FaersFilesCorrections(Processor):
    """Processor that applies corrections to raw FAERS files before ingestion.

    The ``process`` method receives a file path and applies the following
    transformations in order:

    1. **File renaming** -- files listed in ``__FILE_RENAMES`` are physically
       renamed on disk (e.g. ``demo18q1_new.txt`` -> ``demo18q1.txt``).
    2. **Demo header normalisation** -- in all ``demo`` files the column
       ``gndr_cod`` is replaced with ``sex`` in the header row.
    3. **Encoding cleanup** -- files listed in ``__FILES_WITH_ENCODING_ISSUES``
       are re-read with ``errors="ignore"`` and written back, silently
       dropping any characters that cannot be decoded.
    4. **Line-level content fixes** -- files listed in ``__REPLACEMENTS``
       have specific lines patched.  Two categories of fixes exist:
       * *Missing newlines*: two records concatenated on a single line are
         split into separate rows (affects ``drug11q2``, ``drug11q3``,
         ``drug11q4``).
       * *Header typos*: column names are corrected to match the expected
         schema (e.g. ``$ rept_dt$`` -> ``$rept_dt$``,
         ``$lot_nbr$`` -> ``$lot_num$``, ``$outc_code`` -> ``$outc_cod``).
    5. **Metadata enrichment** -- the corrected file is loaded into a pandas
       DataFrame and two columns are appended:
       * ``processing_time``: UTC epoch timestamp (int) of the processing run.
       * ``period``: the FAERS period string derived from the filename
         (e.g. ``"11q2"``).
       The DataFrame is then written back to the same file path.

    The method returns the number of corrections applied (int).
    Files whose first four characters do not match a known FAERS table name
    are left untouched and the method returns ``0``.
    """

    CONTEXT_KEY_FILE = "input_file"
    CONTEXT_KEY_OUTPUT_FOLDER = "output_folder"

    __FILES_WITH_ENCODING_ISSUES: ClassVar[list[str]] = ["drug19q3.txt"]
    __REPLACEMENTS: ClassVar[dict[str, list[dict[str, Any]]]] = {
        "demo12q1.txt": [{"line": 105916, "replacement": ("$E2B0000000182", "E2B0000000182")}],
        "drug11q2.txt": [
            {
                "line": 322966,
                "replacement": (
                    "7475791$1016572493$SS$BLEOMYCIN SULFATE$1$INTRAVENOUS$10 MG/M2 MILLIGRAM(S)/SQ. METER, DAY 1 AND 15, EVERY 28 DAYS, INTRAVENOUS (NOT OTHERWISE SPECIFIED)$$$$$$7475791$1016572490$SS$DOXORUBICIN (DOXORUBICIN) (INJECTION)$2$INTRAVENOUS$25 MG/M2 MILLIGRAM(S)/SQ. METER, DAY 1 AND 15, EVERY 28 DAYS, INTRAVENOUS (NOT OTHERWISE SPECIFIED)$$$$$$",
                    "7475791$1016572493$SS$BLEOMYCIN SULFATE$1$INTRAVENOUS$10 MG/M2 MILLIGRAM(S)/SQ. METER, DAY 1 AND 15, EVERY 28 DAYS, INTRAVENOUS (NOT OTHERWISE SPECIFIED)$$$$$$\n7475791$1016572490$SS$DOXORUBICIN (DOXORUBICIN) (INJECTION)$2$INTRAVENOUS$25 MG/M2 MILLIGRAM(S)/SQ. METER, DAY 1 AND 15, EVERY 28 DAYS, INTRAVENOUS (NOT OTHERWISE SPECIFIED)$$$$$$",
                ),
            }
        ],
        "drug11q3.txt": [
            {
                "line": 247895,
                "replacement": (
                    "7652730$1017185838$PS$FLUOROURACIL$1$$10140 MG$$$$$$7652730$1017255397$SS$BEVACIZUMAB (RHUMAB VEGF)$2$$920 MG$$$$$$",
                    "7652730$1017185838$PS$FLUOROURACIL$1$$10140 MG$$$$$$\n7652730$1017255397$SS$BEVACIZUMAB (RHUMAB VEGF)$2$$920 MG$$$$$$",
                ),
            }
        ],
        "drug11q4.txt": [
            {
                "line": 446737,
                "replacement": (
                    "7941354$1018142410$PS$MEMANTINE HYDROCHLORIDE$1$ORAL$5 MG (5 MG, 1 IN 1 D),ORAL , 105 MG (10 MG, 1 IN 1 D),ORAL$D$D$$$021487$7941354$1018188213$SS$MEMANTINE HYDROCHLORIDE$1$ORAL$15 MG (15 MG, 1 IN 1 D),ORAL$D$D$$$$",
                    "7941354$1018142410$PS$MEMANTINE HYDROCHLORIDE$1$ORAL$5 MG (5 MG, 1 IN 1 D),ORAL , 105 MG (10 MG, 1 IN 1 D),ORAL$D$D$$$021487$\n7941354$1018188213$SS$MEMANTINE HYDROCHLORIDE$1$ORAL$15 MG (15 MG, 1 IN 1 D),ORAL$D$D$$$$",
                ),
            }
        ],
        "demo12q4.txt": [{"line": 0, "replacement": ("$ rept_dt$", "$rept_dt$")}],
        "demo13q1.txt": [{"line": 0, "replacement": ("$ rept_dt$", "$rept_dt$")}],
        "drug12q4.txt": [{"line": 0, "replacement": ("$lot_nbr$", "$lot_num$")}],
        "outc12q4.txt": [{"line": 0, "replacement": ("$outc_code", "$outc_cod")}],
    }
    __FILE_RENAMES: ClassVar[dict[str, str]] = {"demo18q1_new.txt": "demo18q1.txt"}

    __SCHEMAS: ClassVar[dict[str, dict[str, str]]] = {
        "demo": {
            "primaryid": "Int64",
            "caseid": "Int64",
            "caseversion": "Int64",
            "i_f_cod": "object",
            "event_dt": "Int64",
            "mfr_dt": "Int64",
            "init_fda_dt": "Int64",
            "fda_dt": "Int64",
            "rept_cod": "object",
            "auth_num": "object",
            "mfr_num": "object",
            "mfr_sndr": "object",
            "lit_ref": "object",
            "age": "Int64",
            "age_cod": "object",
            "age_grp": "object",
            "sex": "object",
            "e_sub": "object",
            "wt": "Float64",
            "wt_cod": "object",
            "rept_dt": "Int64",
            "to_mfr": "object",
            "occp_cod": "object",
            "reporter_country": "object",
            "occr_country": "object",
            "processing_time": "Int64",
            "period": "object",
        },
        "drug": {
            "primaryid": "Int64",
            "caseid": "Int64",
            "drug_seq": "Int64",
            "role_cod": "object",
            "drugname": "object",
            "prod_ai": "object",
            "val_vbm": "Int64",
            "route": "object",
            "dose_vbm": "object",
            "cum_dose_chr": "object",
            "cum_dose_unit": "object",
            "dechal": "object",
            "rechal": "object",
            "lot_num": "object",
            "exp_dt": "Int64",
            "nda_num": "Float64",
            "dose_amt": "object",
            "dose_unit": "object",
            "dose_form": "object",
            "dose_freq": "object",
            "processing_time": "Int64",
            "period": "object",
        },
        "indi": {
            "primaryid": "Int64",
            "caseid": "Int64",
            "indi_drug_seq": "Int64",
            "indi_pt": "object",
            "processing_time": "Int64",
            "period": "object",
        },
        "outc": {
            "primaryid": "Int64",
            "caseid": "Int64",
            "outc_cod": "object",
            "processing_time": "Int64",
            "period": "object",
        },
        "reac": {
            "primaryid": "Int64",
            "caseid": "Int64",
            "pt": "object",
            "drug_rec_act": "object",
            "processing_time": "Int64",
            "period": "object",
        },
        "rpsr": {
            "primaryid": "Int64",
            "caseid": "Int64",
            "rpsr_cod": "object",
            "processing_time": "Int64",
            "period": "object",
        },
        "ther": {
            "primaryid": "Int64",
            "caseid": "Int64",
            "dsg_drug_seq": "Int64",
            "start_dt": "Int64",
            "end_dt": "Int64",
            "dur": "Float64",
            "dur_cod": "object",
            "processing_time": "Int64",
            "period": "object",
        },
    }

    __COLUMN_DROPS: ClassVar[dict[str, list[str]]] = {"demo": ["foll_seq", "image", "death_dt", "confid"]}
    __COLUMN_RENAMES: ClassVar[dict[str, dict[str, str]]] = {
        "demo": {"isr": "primaryid", "case": "caseid", "i_f_code": "i_f_cod", "gndr_cod": "sex"},
        "drug": {"isr": "primaryid", "case": "caseid"},
        "indi": {"isr": "primaryid", "drug_seq": "indi_drug_seq"},
        "outc": {"isr": "primaryid"},
        "reac": {"isr": "primaryid"},
        "rpsr": {"isr": "primaryid"},
        "ther": {"isr": "primaryid", "drug_seq": "dsg_drug_seq"},
    }

    __COLUMN_ADDS: ClassVar[dict[str, dict[str, None]]] = {
        "demo": {
            "caseversion": None,
            "caseid": None,
            "init_fda_dt": None,
            "auth_num": None,
            "lit_ref": None,
            "age_grp": None,
            "reporter_country": None,
            "occr_country": None,
        },
        "drug": {
            "prod_ai": None,
            "caseid": None,
            "cum_dose_chr": None,
            "cum_dose_unit": None,
            "dose_amt": None,
            "dose_unit": None,
            "dose_form": None,
            "dose_freq": None,
        },
        "indi": {
            "caseid": None,
        },
        "outc": {"caseid": None},
        "reac": {
            "caseid": None,
            "drug_rec_act": None,
        },
        "rpsr": {"caseid": None},
        "ther": {"caseid": None},
    }

    __COLUMN_CASTS: ClassVar[dict[str, Any]] = {
        "age": pd.Int64Dtype,
        "age_cod": str,
        "age_grp": str,
        "auth_num": str,
        "caseid": pd.Int64Dtype,
        "caseversion": pd.Int64Dtype,
        "concept_name": str,
        "cum_dose_unit": str,
        "cum_dose_chr": str,
        "dechal": str,
        "dose_amt": str,
        "dose_form": str,
        "dose_freq": str,
        "dose_unit": str,
        "dose_vbm": str,
        "drug_rec_act": str,
        "drug_seq": pd.Int64Dtype,
        "drugname": str,
        "dsg_drug_seq": pd.Int64Dtype,
        "dur": pd.Float64Dtype,
        "dur_cod": str,
        "e_sub": str,
        "end_dt": pd.Int64Dtype,
        "event_dt": pd.Int64Dtype,
        "exp_dt": pd.Int64Dtype,
        "fda_dt": pd.Int64Dtype,
        "i_f_cod": str,
        "indi_drug_seq": pd.Int64Dtype,
        "indi_pt": str,
        "indi_pt_code": pd.Int64Dtype,
        "indi_pt_soc_code": pd.Int64Dtype,
        "init_fda_dt": pd.Int64Dtype,
        "lit_ref": str,
        "lot_num": str,
        "mfr_dt": pd.Int64Dtype,
        "mfr_num": str,
        "mfr_sndr": str,
        "nda_num": pd.Float64Dtype,
        "occp_cod": str,
        "occr_country": str,
        "outc_cod": str,
        "period": str,
        "primaryid": pd.Int64Dtype,
        "processing_time": pd.Int64Dtype,
        "prod_ai": str,
        "pt": str,
        "pt_code": pd.Int64Dtype,
        "pt_soc_code": pd.Int64Dtype,
        "rechal": str,
        "reporter_country": str,
        "rept_cod": str,
        "rept_dt": pd.Int64Dtype,
        "role_cod": str,
        "route": str,
        "rpsr_cod": str,
        "sex": str,
        "start_dt": pd.Int64Dtype,
        "to_mfr": str,
        "val_vbm": pd.Int64Dtype,
        "wt": pd.Float64Dtype,
        "wt_cod": str,
    }

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialise the processor and record the current UTC timestamp."""
        logger.info(f"[__init__|in] ({config})")

        super().__init__(config=config)
        self._ts = int(datetime.now(tz=timezone.utc).timestamp())  # noqa: UP017
        self._constants = Constants()
        logger.info("[__init__|out]")

    def process(self, context: dict[str, Any] | None = None) -> Any:
        """
        Read a FAERS file and apply configured fixes.

        This includes operations such as file renaming, line/header updates,
        and encoding cleanup. Then load the file as a dataframe, apply further
        processing, validate the final schema, and save the file back.
        The operation is meant to be idempotent.
        """
        logger.info(f"[process|in] ({context})")

        corrections_applied: int = 0

        if context is None or self.CONTEXT_KEY_FILE not in context:
            raise Exception(f"you must provide context for {self.CONTEXT_KEY_FILE}")  # noqa: TRY002

        if self.CONTEXT_KEY_OUTPUT_FOLDER not in context:
            raise Exception(f"you must provide context for {self.CONTEXT_KEY_OUTPUT_FOLDER}")  # noqa: TRY002

        output_folder = context[self.CONTEXT_KEY_OUTPUT_FOLDER]
        Path(output_folder).mkdir(parents=True, exist_ok=True)

        filepath = context[self.CONTEXT_KEY_FILE]
        file = Path(filepath).name
        file_folder = str(Path(filepath).parent)

        file_pattern = r".*([0-9][0-9][Q,q][0-9])\.txt"
        table = file[0:4]

        if table in self._constants.TABLES:
            # process file renaming, corrections and move it to the output folder
            if file in self.__FILE_RENAMES:
                file = self.__FILE_RENAMES[file]
                f_after = str(Path(file_folder) / file)
                logger.debug(f"[__fix_file] renaming {filepath} to {f_after}")
                Path(filepath).rename(f_after)
                filepath = f_after
                file_folder = str(Path(filepath).parent)
                corrections_applied += 1

            if table == "demo":
                self.__change_line(filepath, 0, ("$gndr_cod$", "$sex$"))

            pattern_match = re.search(file_pattern, file)
            if pattern_match is not None:
                # extract period from filename, to be added as metadata later
                group = pattern_match.group(1).lower()
                fileparts = group.split("q")
                year = int(str(datetime.now(tz=timezone.utc).year)[:2] + fileparts[0])  # noqa: UP017
                quarter = int(fileparts[1])
                period = FaersPeriod(year, quarter)

                # process file encoding exceptions found before
                if file in self.__FILES_WITH_ENCODING_ISSUES:
                    self.__restore_file_ignoring_encoding_issues(filepath)
                    corrections_applied += 1

                # process file content replacements
                if file in self.__REPLACEMENTS:
                    changes = self.__REPLACEMENTS[file]
                    for change in changes:
                        line_num = change["line"]
                        replacement = change["replacement"]
                        self.__change_line(filepath, line_num, replacement)
                        corrections_applied += 1

                # add metadata
                df: pd.DataFrame = pd.read_csv(
                    filepath, delimiter=self._constants.CSV_DELIMITER, index_col=False, low_memory=False
                )
                df["processing_time"] = self._ts
                df["period"] = str(period)

                # do further processing as a dataframe
                df, df_corrections = self.__dataframe_processing(table, df)
                corrections_applied += df_corrections

                # validate the end result schema
                self.__validate_schema(table, df)

                # save the corrected file to the output location
                output_file = Path(output_folder) / file
                df.to_csv(output_file, sep=self._constants.CSV_DELIMITER, index=False)

        logger.info(f"[process|out] => corrections_applied: {corrections_applied}")
        return corrections_applied

    def __change_line(self, file_name, line_num, replace: tuple[str, str]) -> None:
        """
        helper method to replace a specific string in a line in a file
        """
        logger.debug(f"[change_line|in] ({file_name}, {line_num}, {replace})")
        with Path(file_name).open() as f:
            lines = f.readlines()
        logger.debug(f"[change_line] original line: {lines[line_num]}")
        line = lines[line_num]
        lines[line_num] = line.replace(replace[0], replace[1])
        with Path(file_name).open("w") as f:
            f.writelines(lines)
        with Path(file_name).open() as f:
            lines = f.readlines()
        logger.debug("[change_line] new lines:")
        logger.debug(f"{lines[line_num]}")
        logger.debug(f"{lines[line_num + 1]}")
        logger.debug("[change_line|out]")

    def __restore_file_ignoring_encoding_issues(self, filename) -> None:
        """
        resaves file back to the fs ignoring encoding issues, basically dropping lines with encoding errors
        """
        logger.debug(f"[restore_file_ignoring_encoding_issues|in] ({filename})")
        with Path(filename).open(errors="ignore") as f:
            lines = f.readlines()
        with Path(filename).open("w") as f:
            f.writelines(lines)
        logger.debug("[restore_file_ignoring_encoding_issues|out]")

    def __validate_schema(self, table: str, df: pd.DataFrame) -> None:
        """
        Validate a dataframe against its expected schema.

        Args:
            table: Table name to look up schema definition
            df: pandas DataFrame to validate

        Raises:
            Exception: If schema validation fails for the table
        """
        logger.debug(f"[__validate_schema|in] ({table}, {df.shape})")

        if table in self.__SCHEMAS:
            expected_schema = self.__SCHEMAS[table]
            self.__validate_expectations(df, expected_schema)

        logger.debug("[__validate_schema|out]")

    def __dataframe_processing(self, table: str, df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        """
        Process and standardize a dataframe by lowercasing columns, dropping unnecessary columns,
        renaming columns, and adding missing columns based on table type.

        Args:
            table: Table name (e.g., 'demo', 'drug', 'indi', 'outc', 'reac', 'rpsr', 'ther')
            df: Input pandas DataFrame to process

        Returns:
            Processed pandas DataFrame with standardized columns
        """
        logger.debug(f"[dataframe_processing|in] ({table}, {df.shape})")

        corrections_applied: int = 0

        df.columns = df.columns.str.lower()
        logger.debug(f"[dataframe_processing] lower-cased table {table} columns")

        if table in self.__COLUMN_DROPS:
            for col in self.__COLUMN_DROPS[table]:
                if col in df.columns:
                    del df[col]
                    logger.debug(f"[dataframe_processing] table {table} column dropped: {col}")
                    corrections_applied += 1

        if table in self.__COLUMN_RENAMES:
            df = df.rename(columns=self.__COLUMN_RENAMES[table])  # noqa: PD901
            logger.debug(f"[dataframe_processing] table {table} columns renamed")
            corrections_applied += 1

        if table in self.__COLUMN_ADDS:
            for col, default in self.__COLUMN_ADDS[table].items():
                df[col] = default
                logger.debug(f"[dataframe_processing] table {table} column added: {col}")
                corrections_applied += 1

        for col, _dtype in self.__COLUMN_CASTS.items():
            if col in df.columns:
                if _dtype == pd.Int64Dtype:
                    df[col] = pd.to_numeric(df[col], errors="coerce").round(0).astype(_dtype())
                elif _dtype == pd.Float64Dtype:
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype(_dtype())
                else:
                    df[col] = df[col].astype(_dtype, errors="raise")
                logger.debug(f"[dataframe_processing] table {table} column casted: {col} to {_dtype}")

        logger.debug(f"[dataframe_processing|out] => {df.shape}, {corrections_applied}")
        return df, corrections_applied

    def __validate_expectations(self, df, expected: dict) -> None:
        """Validate that DataFrame schema matches expected column types.

        Args:
            df: DataFrame to validate.
            expected: Dict mapping column names to expected dtype strings.

        Raises:
            ValueError: If schema has missing columns or type mismatches.
        """
        actual = df.dtypes.astype(str).to_dict()
        mismatches = {col: (expected[col], actual.get(col)) for col in expected if actual.get(col) != expected[col]}
        missing = [col for col in expected if col not in actual]
        if missing or mismatches:
            raise ValueError(f"Schema mismatch — missing: {missing}, wrong types: {mismatches}")


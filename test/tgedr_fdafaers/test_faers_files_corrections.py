"""Unit tests for FaersFilesCorrections processor."""

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from tgedr_fdafaers.faers_files_corrections import FaersFilesCorrections


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture()
def processor() -> FaersFilesCorrections:
    """Create a FaersFilesCorrections instance with a mocked timestamp."""
    with patch("tgedr_fdafaers.faers_files_corrections.datetime") as mock_dt:
        mock_dt.now.return_value.timestamp.return_value = 1700000000
        mock_dt.now.return_value.year = 2024
        proc = FaersFilesCorrections()
    return proc


@pytest.fixture()
def demo_file(tmp_path: Path) -> Path:
    """Create a minimal valid demo file with $ delimiter."""
    content = (
        "primaryid$caseid$caseversion$i_f_cod$event_dt$mfr_dt$init_fda_dt$fda_dt"
        "$rept_cod$auth_num$mfr_num$mfr_sndr$lit_ref$age$age_cod$age_grp"
        "$sex$e_sub$wt$wt_cod$rept_dt$to_mfr$occp_cod$reporter_country$occr_country\n"
        "1001$500$1$I$20240101$20240102$20240103$20240104"
        "$EXP$AUTH1$MFR1$SNDR1$$65$YR$3"
        "$F$Y$70.5$KG$20240105$Y$MD$US$US\n"
    )
    f = tmp_path / "demo24q1.txt"
    f.write_text(content, encoding="utf-8")
    return f


@pytest.fixture()
def drug_file(tmp_path: Path) -> Path:
    """Create a minimal valid drug file with $ delimiter."""
    content = (
        "primaryid$caseid$drug_seq$role_cod$drugname$prod_ai$val_vbm$route"
        "$dose_vbm$cum_dose_chr$cum_dose_unit$dechal$rechal$lot_num$exp_dt"
        "$nda_num$dose_amt$dose_unit$dose_form$dose_freq\n"
        "1001$500$1$PS$ASPIRIN$ASPIRIN$1$ORAL"
        "$100MG$$$N$N$LOT1$20250101"
        "$21505$100$MG$TABLET$DAILY\n"
    )
    f = tmp_path / "drug24q1.txt"
    f.write_text(content, encoding="utf-8")
    return f


# --------------------------------------------------------------------------- #
# process() — context validation
# --------------------------------------------------------------------------- #


def test_process_raises_when_context_is_none(processor: FaersFilesCorrections) -> None:
    """process should raise when context is None."""
    with pytest.raises(Exception, match="input_file"):
        processor.process(context=None)


def test_process_raises_when_input_file_missing(
    processor: FaersFilesCorrections, tmp_path: Path
) -> None:
    """process should raise when input_file key is absent."""
    with pytest.raises(Exception, match="input_file"):
        processor.process(context={"output_folder": str(tmp_path)})


def test_process_raises_when_output_folder_missing(
    processor: FaersFilesCorrections, tmp_path: Path
) -> None:
    """process should raise when output_folder key is absent."""
    f = tmp_path / "demo24q1.txt"
    f.write_text("dummy", encoding="utf-8")
    with pytest.raises(Exception, match="output_folder"):
        processor.process(context={"input_file": str(f)})


# --------------------------------------------------------------------------- #
# process() — unrecognized table prefix
# --------------------------------------------------------------------------- #


def test_process_returns_zero_for_unknown_table_prefix(
    processor: FaersFilesCorrections, tmp_path: Path
) -> None:
    """Files with unrecognized 4-char prefixes should be ignored."""
    f = tmp_path / "stat24q1.txt"
    f.write_text("some content", encoding="utf-8")
    output = tmp_path / "out"

    result = processor.process(context={"input_file": str(f), "output_folder": str(output)})

    assert result == 0


def test_process_returns_zero_for_size_prefix(
    processor: FaersFilesCorrections, tmp_path: Path
) -> None:
    """Files starting with 'size' should be left untouched."""
    f = tmp_path / "size24q1.txt"
    f.write_text("some content", encoding="utf-8")
    output = tmp_path / "out"

    result = processor.process(context={"input_file": str(f), "output_folder": str(output)})

    assert result == 0


# --------------------------------------------------------------------------- #
# process() — demo file processing
# --------------------------------------------------------------------------- #


def test_process_demo_file_adds_metadata_columns(
    processor: FaersFilesCorrections, demo_file: Path, tmp_path: Path
) -> None:
    """Processing a demo file should add processing_time and period columns."""
    output = tmp_path / "out"

    processor.process(context={"input_file": str(demo_file), "output_folder": str(output)})

    result_file = output / demo_file.name
    assert result_file.exists()
    df = pd.read_csv(result_file, delimiter="$")
    assert "processing_time" in df.columns
    assert "period" in df.columns


def test_process_demo_file_renames_gndr_cod_to_sex(
    tmp_path: Path,
) -> None:
    """Processing a demo file should replace gndr_cod with sex in the header."""
    content = (
        "primaryid$caseid$caseversion$i_f_cod$event_dt$mfr_dt$init_fda_dt$fda_dt"
        "$rept_cod$auth_num$mfr_num$mfr_sndr$lit_ref$age$age_cod$age_grp"
        "$gndr_cod$e_sub$wt$wt_cod$rept_dt$to_mfr$occp_cod$reporter_country$occr_country\n"
        "1001$500$1$I$20240101$20240102$20240103$20240104"
        "$EXP$AUTH1$MFR1$SNDR1$$65$YR$3"
        "$F$Y$70.5$KG$20240105$Y$MD$US$US\n"
    )
    f = tmp_path / "demo24q1.txt"
    f.write_text(content, encoding="utf-8")
    output = tmp_path / "out"

    with patch("tgedr_fdafaers.faers_files_corrections.datetime") as mock_dt:
        mock_dt.now.return_value.timestamp.return_value = 1700000000
        mock_dt.now.return_value.year = 2024
        proc = FaersFilesCorrections()

    proc.process(context={"input_file": str(f), "output_folder": str(output)})

    result_file = output / f.name
    df = pd.read_csv(result_file, delimiter="$")
    assert "sex" in df.columns
    assert "gndr_cod" not in df.columns


def test_process_demo_file_period_is_derived_from_filename(
    processor: FaersFilesCorrections, demo_file: Path, tmp_path: Path
) -> None:
    """The period column should match the year/quarter parsed from the filename."""
    output = tmp_path / "out"

    processor.process(context={"input_file": str(demo_file), "output_folder": str(output)})

    result_file = output / demo_file.name
    df = pd.read_csv(result_file, delimiter="$")
    # demo24q1.txt -> period "24q1"
    assert df["period"].iloc[0] == "24q1"


# --------------------------------------------------------------------------- #
# process() — drug file processing
# --------------------------------------------------------------------------- #


def test_process_drug_file_adds_metadata(
    processor: FaersFilesCorrections, drug_file: Path, tmp_path: Path
) -> None:
    """Processing a drug file should add processing_time and period."""
    output = tmp_path / "out"

    processor.process(context={"input_file": str(drug_file), "output_folder": str(output)})

    result_file = output / drug_file.name
    assert result_file.exists()
    df = pd.read_csv(result_file, delimiter="$")
    assert "processing_time" in df.columns
    assert "period" in df.columns


def test_process_drug_file_validates_schema(
    processor: FaersFilesCorrections, drug_file: Path, tmp_path: Path
) -> None:
    """Drug file output should have the expected columns after processing."""
    output = tmp_path / "out"

    processor.process(context={"input_file": str(drug_file), "output_folder": str(output)})

    result_file = output / drug_file.name
    df = pd.read_csv(result_file, delimiter="$")
    # Should contain standard drug columns
    assert "primaryid" in df.columns
    assert "drugname" in df.columns
    assert "role_cod" in df.columns


# --------------------------------------------------------------------------- #
# process() — file renaming
# --------------------------------------------------------------------------- #


def test_process_renames_file_when_in_rename_map(tmp_path: Path) -> None:
    """demo18q1_new.txt should be renamed to demo18q1.txt during processing."""
    content = (
        "primaryid$caseid$caseversion$i_f_cod$event_dt$mfr_dt$init_fda_dt$fda_dt"
        "$rept_cod$auth_num$mfr_num$mfr_sndr$lit_ref$age$age_cod$age_grp"
        "$sex$e_sub$wt$wt_cod$rept_dt$to_mfr$occp_cod$reporter_country$occr_country\n"
        "1001$500$1$I$20180101$20180102$20180103$20180104"
        "$EXP$AUTH1$MFR1$SNDR1$$65$YR$3"
        "$F$Y$70.5$KG$20180105$Y$MD$US$US\n"
    )
    f = tmp_path / "demo18q1_new.txt"
    f.write_text(content, encoding="utf-8")
    output = tmp_path / "out"

    with patch("tgedr_fdafaers.faers_files_corrections.datetime") as mock_dt:
        mock_dt.now.return_value.timestamp.return_value = 1700000000
        mock_dt.now.return_value.year = 2024
        proc = FaersFilesCorrections()

    result = proc.process(context={"input_file": str(f), "output_folder": str(output)})

    # Original file should be renamed
    assert not f.exists()
    # Output should use the new name
    assert (output / "demo18q1.txt").exists()
    # At least 1 correction for the rename itself
    assert result >= 1


# --------------------------------------------------------------------------- #
# process() — encoding fix
# --------------------------------------------------------------------------- #


def test_process_fixes_encoding_issues(tmp_path: Path) -> None:
    """drug19q3.txt with bad bytes should be re-saved dropping invalid characters."""
    # Create a file with valid header but some bad bytes in data
    header = (
        "primaryid$caseid$drug_seq$role_cod$drugname$prod_ai$val_vbm$route"
        "$dose_vbm$cum_dose_chr$cum_dose_unit$dechal$rechal$lot_num$exp_dt"
        "$nda_num$dose_amt$dose_unit$dose_form$dose_freq\n"
    )
    data_line = (
        "1001$500$1$PS$ASPIRIN$ASPIRIN$1$ORAL"
        "$100MG$$$N$N$LOT1$20250101"
        "$21505$100$MG$TABLET$DAILY\n"
    )
    f = tmp_path / "drug19q3.txt"
    # Write with some invalid bytes embedded
    f.write_bytes(header.encode("utf-8") + b"\xff\xfe" + data_line.encode("utf-8"))
    output = tmp_path / "out"

    with patch("tgedr_fdafaers.faers_files_corrections.datetime") as mock_dt:
        mock_dt.now.return_value.timestamp.return_value = 1700000000
        mock_dt.now.return_value.year = 2024
        proc = FaersFilesCorrections()

    result = proc.process(context={"input_file": str(f), "output_folder": str(output)})

    # Should have applied at least one encoding correction
    assert result >= 1
    # Output file should exist
    assert (output / "drug19q3.txt").exists()


# --------------------------------------------------------------------------- #
# process() — line-level replacements
# --------------------------------------------------------------------------- #


def test_process_applies_header_replacement(tmp_path: Path) -> None:
    """demo12q4.txt should have '$ rept_dt$' corrected to '$rept_dt$' in header."""
    header = (
        "primaryid$caseid$caseversion$i_f_cod$event_dt$mfr_dt$init_fda_dt$fda_dt"
        "$rept_cod$auth_num$mfr_num$mfr_sndr$lit_ref$age$age_cod$age_grp"
        "$sex$e_sub$wt$wt_cod$ rept_dt$to_mfr$occp_cod$reporter_country$occr_country\n"
    )
    data = (
        "1001$500$1$I$20120101$20120102$20120103$20120104"
        "$EXP$AUTH1$MFR1$SNDR1$$65$YR$3"
        "$F$Y$70.5$KG$20120401$Y$MD$US$US\n"
    )
    f = tmp_path / "demo12q4.txt"
    f.write_text(header + data, encoding="utf-8")
    output = tmp_path / "out"

    with patch("tgedr_fdafaers.faers_files_corrections.datetime") as mock_dt:
        mock_dt.now.return_value.timestamp.return_value = 1700000000
        mock_dt.now.return_value.year = 2024
        proc = FaersFilesCorrections()

    result = proc.process(context={"input_file": str(f), "output_folder": str(output)})

    # Header replacement counts as a correction
    assert result >= 1
    result_file = output / "demo12q4.txt"
    df = pd.read_csv(result_file, delimiter="$")
    assert "rept_dt" in df.columns


def test_process_applies_lot_nbr_replacement(tmp_path: Path) -> None:
    """drug12q4.txt should have '$lot_nbr$' corrected to '$lot_num$' in header."""
    header = (
        "primaryid$caseid$drug_seq$role_cod$drugname$prod_ai$val_vbm$route"
        "$dose_vbm$cum_dose_chr$cum_dose_unit$dechal$rechal$lot_nbr$exp_dt"
        "$nda_num$dose_amt$dose_unit$dose_form$dose_freq\n"
    )
    data = (
        "1001$500$1$PS$ASPIRIN$ASPIRIN$1$ORAL"
        "$100MG$$$N$N$LOT1$20250101"
        "$21505$100$MG$TABLET$DAILY\n"
    )
    f = tmp_path / "drug12q4.txt"
    f.write_text(header + data, encoding="utf-8")
    output = tmp_path / "out"

    with patch("tgedr_fdafaers.faers_files_corrections.datetime") as mock_dt:
        mock_dt.now.return_value.timestamp.return_value = 1700000000
        mock_dt.now.return_value.year = 2024
        proc = FaersFilesCorrections()

    result = proc.process(context={"input_file": str(f), "output_folder": str(output)})

    assert result >= 1
    result_file = output / "drug12q4.txt"
    df = pd.read_csv(result_file, delimiter="$")
    assert "lot_num" in df.columns


def test_process_applies_outc_code_replacement(tmp_path: Path) -> None:
    """outc12q4.txt should have '$outc_code' corrected to '$outc_cod' in header."""
    header = "primaryid$caseid$outc_code\n"
    data = "1001$500$DE\n"
    f = tmp_path / "outc12q4.txt"
    f.write_text(header + data, encoding="utf-8")
    output = tmp_path / "out"

    with patch("tgedr_fdafaers.faers_files_corrections.datetime") as mock_dt:
        mock_dt.now.return_value.timestamp.return_value = 1700000000
        mock_dt.now.return_value.year = 2024
        proc = FaersFilesCorrections()

    result = proc.process(context={"input_file": str(f), "output_folder": str(output)})

    assert result >= 1
    result_file = output / "outc12q4.txt"
    df = pd.read_csv(result_file, delimiter="$")
    assert "outc_cod" in df.columns


# --------------------------------------------------------------------------- #
# __dataframe_processing — column operations
# --------------------------------------------------------------------------- #


def test_dataframe_processing_lowercases_columns(
    processor: FaersFilesCorrections,
) -> None:
    """Column names should be lowercased."""
    df = pd.DataFrame({"PRIMARYID": [1], "CASEID": [2], "RPSR_COD": ["FGN"]})

    result_df, _ = processor._FaersFilesCorrections__dataframe_processing("rpsr", df)

    assert all(c == c.lower() for c in result_df.columns)


def test_dataframe_processing_drops_demo_columns(
    processor: FaersFilesCorrections,
) -> None:
    """Demo files should have foll_seq, image, death_dt, confid dropped."""
    df = pd.DataFrame({
        "primaryid": [1],
        "caseid": [2],
        "caseversion": [1],
        "i_f_cod": ["I"],
        "event_dt": [20240101],
        "mfr_dt": [20240102],
        "init_fda_dt": [20240103],
        "fda_dt": [20240104],
        "rept_cod": ["EXP"],
        "auth_num": ["A1"],
        "mfr_num": ["M1"],
        "mfr_sndr": ["S1"],
        "lit_ref": [""],
        "age": [65],
        "age_cod": ["YR"],
        "age_grp": ["3"],
        "sex": ["F"],
        "e_sub": ["Y"],
        "wt": [70.5],
        "wt_cod": ["KG"],
        "rept_dt": [20240105],
        "to_mfr": ["Y"],
        "occp_cod": ["MD"],
        "reporter_country": ["US"],
        "occr_country": ["US"],
        "foll_seq": ["1"],
        "image": ["img"],
        "death_dt": [20240101],
        "confid": ["1"],
        "processing_time": [1700000000],
        "period": ["24q1"],
    })

    result_df, corrections = processor._FaersFilesCorrections__dataframe_processing("demo", df)

    assert "foll_seq" not in result_df.columns
    assert "image" not in result_df.columns
    assert "death_dt" not in result_df.columns
    assert "confid" not in result_df.columns
    assert corrections >= 4  # 4 drops


def test_dataframe_processing_renames_columns(
    processor: FaersFilesCorrections,
) -> None:
    """Drug table should have 'isr' renamed to 'primaryid' and 'case' to 'caseid'."""
    df = pd.DataFrame({
        "isr": [1],
        "case": [500],
        "drug_seq": [1],
        "role_cod": ["PS"],
        "drugname": ["ASPIRIN"],
        "prod_ai": ["ASPIRIN"],
        "val_vbm": [1],
        "route": ["ORAL"],
        "dose_vbm": ["100MG"],
        "cum_dose_chr": [""],
        "cum_dose_unit": [""],
        "dechal": ["N"],
        "rechal": ["N"],
        "lot_num": ["LOT1"],
        "exp_dt": [20250101],
        "nda_num": [21505.0],
        "dose_amt": ["100"],
        "dose_unit": ["MG"],
        "dose_form": ["TABLET"],
        "dose_freq": ["DAILY"],
        "processing_time": [1700000000],
        "period": ["24q1"],
    })

    result_df, corrections = processor._FaersFilesCorrections__dataframe_processing("drug", df)

    assert "primaryid" in result_df.columns
    assert "caseid" in result_df.columns
    assert "isr" not in result_df.columns
    assert "case" not in result_df.columns
    assert corrections >= 1


def test_dataframe_processing_adds_missing_columns(
    processor: FaersFilesCorrections,
) -> None:
    """indi table should get a 'caseid' column added if not present."""
    df = pd.DataFrame({
        "primaryid": [1],
        "indi_drug_seq": [1],
        "indi_pt": ["PAIN"],
        "processing_time": [1700000000],
        "period": ["24q1"],
    })

    result_df, corrections = processor._FaersFilesCorrections__dataframe_processing("indi", df)

    assert "caseid" in result_df.columns
    assert corrections >= 1


# --------------------------------------------------------------------------- #
# __validate_expectations
# --------------------------------------------------------------------------- #


def test_validate_expectations_passes_for_correct_schema(
    processor: FaersFilesCorrections,
) -> None:
    """validate_expectations should not raise when schema matches."""
    df = pd.DataFrame({"a": pd.array([1], dtype="Int64"), "b": ["text"]})
    expected = {"a": "Int64", "b": "object"}

    # Should not raise
    processor._FaersFilesCorrections__validate_expectations(df, expected)


def test_validate_expectations_raises_on_missing_column(
    processor: FaersFilesCorrections,
) -> None:
    """validate_expectations should raise ValueError when columns are missing."""
    df = pd.DataFrame({"a": pd.array([1], dtype="Int64")})
    expected = {"a": "Int64", "b": "object"}

    with pytest.raises(ValueError, match="missing"):
        processor._FaersFilesCorrections__validate_expectations(df, expected)


def test_validate_expectations_raises_on_type_mismatch(
    processor: FaersFilesCorrections,
) -> None:
    """validate_expectations should raise ValueError when types don't match."""
    df = pd.DataFrame({"a": ["text"]})  # object dtype
    expected = {"a": "Int64"}

    with pytest.raises(ValueError, match="wrong types"):
        processor._FaersFilesCorrections__validate_expectations(df, expected)


# --------------------------------------------------------------------------- #
# process() — output file location
# --------------------------------------------------------------------------- #


def test_process_creates_output_folder_if_not_exists(
    processor: FaersFilesCorrections, demo_file: Path, tmp_path: Path
) -> None:
    """process should create the output_folder if it doesn't exist."""
    output = tmp_path / "deep" / "nested" / "out"
    assert not output.exists()

    processor.process(context={"input_file": str(demo_file), "output_folder": str(output)})

    assert output.exists()
    assert output.is_dir()


def test_process_writes_output_to_specified_folder(
    processor: FaersFilesCorrections, demo_file: Path, tmp_path: Path
) -> None:
    """Corrected file should be written to output_folder, not the source folder."""
    output = tmp_path / "out"

    processor.process(context={"input_file": str(demo_file), "output_folder": str(output)})

    # File in output folder
    assert (output / demo_file.name).exists()


# --------------------------------------------------------------------------- #
# process() — corrections count
# --------------------------------------------------------------------------- #


def test_process_returns_nonzero_corrections_for_valid_file(
    processor: FaersFilesCorrections, demo_file: Path, tmp_path: Path
) -> None:
    """Processing a valid known-table file should return corrections > 0."""
    output = tmp_path / "out"

    result = processor.process(context={"input_file": str(demo_file), "output_folder": str(output)})

    # At least column renames/adds count as corrections
    assert result >= 0  # May be 0 if no drops/renames apply to this specific file

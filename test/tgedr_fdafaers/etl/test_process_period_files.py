"""Unit tests for the ProcessPeriodFiles ETL."""

import os
from pathlib import Path
from unittest.mock import patch, MagicMock, call

from tgedr_fdafaers.etl.process_period_files import ProcessPeriodFiles


# --------------------------------------------------------------------------- #
# extract
# --------------------------------------------------------------------------- #


@patch("tgedr_fdafaers.etl.process_period_files.UtilsIO.deflate_zip")
@patch("tgedr_fdafaers.etl.process_period_files.UtilsIO.tmp_dir", return_value="/tmp/fake")
def test_extract_deflates_each_file(mock_tmp_dir, mock_deflate) -> None:
    """extract should call deflate_zip for each comma-separated file path."""
    mock_deflate.side_effect = [["/tmp/fake/demo24q1.txt"], ["/tmp/fake/drug24q1.txt"]]

    etl = ProcessPeriodFiles(configuration={"files": "/path/a.zip,/path/b.zip"})
    etl.extract()

    assert mock_deflate.call_count == 2
    assert etl._files_deflated == ["/tmp/fake/demo24q1.txt", "/tmp/fake/drug24q1.txt"]


@patch("tgedr_fdafaers.etl.process_period_files.UtilsIO.deflate_zip")
@patch("tgedr_fdafaers.etl.process_period_files.UtilsIO.tmp_dir", return_value="/tmp/fake")
def test_extract_passes_lower_filename_true(mock_tmp_dir, mock_deflate) -> None:
    """extract should pass lower_filename=True to deflate_zip."""
    mock_deflate.return_value = []

    etl = ProcessPeriodFiles(configuration={"files": "/path/a.zip"})
    etl.extract()

    _, kwargs = mock_deflate.call_args
    assert kwargs.get("lower_filename") is True


@patch("tgedr_fdafaers.etl.process_period_files.UtilsIO.deflate_zip")
@patch("tgedr_fdafaers.etl.process_period_files.UtilsIO.tmp_dir", return_value="/tmp/fake")
def test_extract_file_filter_rejects_stat_and_size_files(mock_tmp_dir, mock_deflate) -> None:
    """The zip filter used by extract should reject size*.txt, stat*.txt, and dotfiles."""
    mock_deflate.return_value = []

    etl = ProcessPeriodFiles(configuration={"files": "/path/a.zip"})
    etl.extract()

    # Get the file_filter function passed to deflate_zip
    _, kwargs = mock_deflate.call_args
    file_filter = kwargs["file_filter"]

    # Should reject these
    assert file_filter("size24q1.txt") is False
    assert file_filter("stat24q1.txt") is False
    assert file_filter(".hidden.txt") is False
    # Should accept these
    assert file_filter("demo24q1.txt") is True
    assert file_filter("drug24q1.txt") is True
    assert file_filter("indi24q1.txt") is True


@patch("tgedr_fdafaers.etl.process_period_files.UtilsIO.deflate_zip")
@patch("tgedr_fdafaers.etl.process_period_files.UtilsIO.tmp_dir", return_value="/tmp/fake")
def test_extract_file_filter_rejects_non_txt_files(mock_tmp_dir, mock_deflate) -> None:
    """The zip filter should reject non-.txt files."""
    mock_deflate.return_value = []

    etl = ProcessPeriodFiles(configuration={"files": "/path/a.zip"})
    etl.extract()

    _, kwargs = mock_deflate.call_args
    file_filter = kwargs["file_filter"]

    assert file_filter("demo24q1.csv") is False
    assert file_filter("readme.md") is False
    assert file_filter("data.pdf") is False


@patch("tgedr_fdafaers.etl.process_period_files.UtilsIO.deflate_zip")
@patch("tgedr_fdafaers.etl.process_period_files.UtilsIO.tmp_dir", return_value="/tmp/fake")
def test_extract_strips_whitespace_from_file_paths(mock_tmp_dir, mock_deflate) -> None:
    """extract should strip whitespace from file paths."""
    mock_deflate.return_value = []

    etl = ProcessPeriodFiles(configuration={"files": " /path/a.zip , /path/b.zip "})
    etl.extract()

    calls = mock_deflate.call_args_list
    assert calls[0][0][0] == "/path/a.zip"
    assert calls[1][0][0] == "/path/b.zip"


# --------------------------------------------------------------------------- #
# transform
# --------------------------------------------------------------------------- #


@patch("tgedr_fdafaers.etl.process_period_files.FaersFilesCorrections")
@patch("tgedr_fdafaers.etl.process_period_files.UtilsIO.tmp_dir", return_value="/tmp/fake")
def test_transform_calls_corrections_for_each_file(mock_tmp_dir, mock_corrections_cls) -> None:
    """transform should call process() on each deflated file."""
    mock_corrections = MagicMock()
    mock_corrections_cls.return_value = mock_corrections

    etl = ProcessPeriodFiles()
    etl._files_deflated = ["/tmp/fake/demo24q1.txt", "/tmp/fake/drug24q1.txt"]

    etl.transform()

    assert mock_corrections.process.call_count == 2
    # Verify context structure
    expected_calls = [
        call(context={"input_file": "/tmp/fake/demo24q1.txt", "output_folder": etl._output_dir}),
        call(context={"input_file": "/tmp/fake/drug24q1.txt", "output_folder": etl._output_dir}),
    ]
    mock_corrections.process.assert_has_calls(expected_calls)


@patch("tgedr_fdafaers.etl.process_period_files.FaersFilesCorrections")
@patch("tgedr_fdafaers.etl.process_period_files.UtilsIO.tmp_dir", return_value="/tmp/fake")
def test_transform_does_nothing_when_no_files_deflated(mock_tmp_dir, mock_corrections_cls) -> None:
    """transform should not instantiate corrections if no files were deflated."""
    etl = ProcessPeriodFiles()
    etl._files_deflated = []

    etl.transform()

    mock_corrections_cls.assert_not_called()


# --------------------------------------------------------------------------- #
# load
# --------------------------------------------------------------------------- #


@patch("tgedr_fdafaers.etl.process_period_files.UtilsIO.tmp_dir")
def test_load_returns_comma_separated_output_files(mock_tmp_dir, tmp_path: Path) -> None:
    """load should return all files in _output_dir as a comma-separated string."""
    mock_tmp_dir.return_value = str(tmp_path / "tmp")
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "demo24q1.txt").write_text("a", encoding="utf-8")
    (output_dir / "drug24q1.txt").write_text("b", encoding="utf-8")

    etl = ProcessPeriodFiles()
    etl._output_dir = str(output_dir)

    result = etl.load()

    # Result should contain both files (order from os.listdir may vary)
    parts = result.split(",")
    assert len(parts) == 2
    filenames = {Path(p).name for p in parts}
    assert filenames == {"demo24q1.txt", "drug24q1.txt"}


@patch("tgedr_fdafaers.etl.process_period_files.UtilsIO.tmp_dir")
def test_load_returns_empty_string_when_output_dir_is_empty(mock_tmp_dir, tmp_path: Path) -> None:
    """load should return empty string when no files are in the output directory."""
    mock_tmp_dir.return_value = str(tmp_path / "tmp")
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    etl = ProcessPeriodFiles()
    etl._output_dir = str(output_dir)

    result = etl.load()

    assert result == ""


# --------------------------------------------------------------------------- #
# end-to-end
# --------------------------------------------------------------------------- #


@patch("tgedr_fdafaers.etl.process_period_files.FaersFilesCorrections")
@patch("tgedr_fdafaers.etl.process_period_files.UtilsIO.deflate_zip")
@patch("tgedr_fdafaers.etl.process_period_files.UtilsIO.tmp_dir")
def test_full_etl_pipeline(mock_tmp_dir, mock_deflate, mock_corrections_cls, tmp_path: Path) -> None:
    """Full pipeline: extract deflates, transform corrects, load returns output paths."""
    tmp_dir = tmp_path / "tmp"
    output_dir = tmp_path / "output"
    tmp_dir.mkdir()
    output_dir.mkdir()

    mock_tmp_dir.side_effect = [str(tmp_dir), str(output_dir)]
    mock_deflate.return_value = [str(tmp_dir / "demo24q1.txt")]

    mock_corrections = MagicMock()
    mock_corrections_cls.return_value = mock_corrections

    # Simulate corrections writing to output dir
    def fake_process(context):
        output_folder = context["output_folder"]
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        (Path(output_folder) / "demo24q1.txt").write_text("corrected", encoding="utf-8")
        return 1

    mock_corrections.process.side_effect = fake_process

    etl = ProcessPeriodFiles(configuration={"files": "/path/archive.zip"})
    etl.extract()
    etl.transform()
    result = etl.load()

    assert "demo24q1.txt" in result
    mock_deflate.assert_called_once()
    mock_corrections.process.assert_called_once()

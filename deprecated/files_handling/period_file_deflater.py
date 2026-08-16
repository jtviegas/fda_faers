"""Processor that extracts FAERS zip archives into individual text files."""

import shutil
from typing import Any
import os
import zipfile
import logging
import tempfile
from tgedr_dataops_abs.processor import Processor


logger = logging.getLogger(__name__)


class PeriodFileDeflater(Processor):
    """Extract FAERS zip files and move the resulting text files to an output folder."""

    CONFIG_KEY_INPUT_FOLDER = "input_folder"
    CONFIG_KEY_OUTPUT_FOLDER = "output_folder"
    CONTEXT_KEY_FILE_FILTER = "file_filter"

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialise with required *input_folder* and *output_folder* config."""
        super().__init__(config=config)
        if (
            not self._config
            or self.CONFIG_KEY_INPUT_FOLDER not in self._config
            or self.CONFIG_KEY_OUTPUT_FOLDER not in self._config
        ):
            raise Exception(
                f"{self.CONFIG_KEY_INPUT_FOLDER} and {self.CONFIG_KEY_OUTPUT_FOLDER} must both be provided in config"
            )
        self._input_folder = self._config[self.CONFIG_KEY_INPUT_FOLDER]
        self._output_folder = self._config[self.CONFIG_KEY_OUTPUT_FOLDER]
        # just to be sure that the output folder exists before we try to move files there
        os.makedirs(self._output_folder, exist_ok=True)

    def process(self, context: dict[str, Any] | None = None) -> list[str]:
        """
        in this case we unzip the files

        Parameters:
            context (dict[str, Any] | None): key-value map with context configuration
        """
        logger.info(f"[process|in] ({context})")

        file_filter: list[str] | None = context.get(self.CONTEXT_KEY_FILE_FILTER, None) if context else None

        result: list[str] = []
        zip_files: list[str] = [
            os.path.join(self._input_folder, file)
            for file in os.listdir(self._input_folder)
            if str.lower(file).endswith(".zip") and ((file_filter is None) or (file in file_filter))
        ]
        for zip_file in zip_files:
            files = self._deflate(zip_file, self._output_folder)
            result.extend(files)

        logger.info(f"[process|out] => {result}")
        return result

    def _deflate(self, file: str, target_folder: str) -> list[str]:
        """
        helper function to deflate the files
        """
        logger.debug(f"[_deflate|in] ({file}, {target_folder})")
        result: list[str] = []

        tmp_folder = tempfile.mkdtemp(dir=(os.path.sep + "tmp"))
        zipdata = zipfile.ZipFile(file)
        zipinfos = zipdata.infolist()
        # iterate through each file
        for zipinfo in zipinfos:
            zipfile_name = os.path.basename(zipinfo.filename)
            zipfile_folder = os.path.dirname(zipinfo.filename)

            if (
                str.lower(zipfile_name).endswith(".txt")
                and str.lower(zipfile_name)[:4] not in ["size", "stat"]
                and zipfile_name[0] != "."
            ):
                zipdata.extract(zipinfo, tmp_folder)
                logger.debug(f"[_deflate] extracting {zipinfo.filename} to {tmp_folder}")
                original_filename = os.path.join(tmp_folder, zipinfo.filename)
                new_filename = os.path.join(tmp_folder, zipfile_folder, zipfile_name.lower())
                os.rename(original_filename, new_filename)
                destination_file = os.path.join(target_folder, zipfile_name.lower())
                if os.path.exists(destination_file):
                    os.remove(destination_file)
                shutil.move(new_filename, target_folder)
                result.append(destination_file)

        logger.debug(f"[_deflate|out] => {result}")
        return result

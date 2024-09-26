# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""File containing function for FTaaS data import component."""

import json
import logging
import random
import argparse
from argparse import Namespace
from pathlib import Path

from azureml.acft.contrib.hf import VERSION, PROJECT_NAME
from azureml.acft.contrib.hf.nlp.constants.constants import (
    LOGS_TO_BE_FILTERED_IN_APPINSIGHTS,
)
from azureml.acft.common_components import (
    get_logger_app,
    set_logging_parameters,
    LoggingLiterals,
)
from azureml.acft.common_components.utils.error_handling.swallow_all_exceptions_decorator import (
    swallow_all_exceptions,
)
from azureml.telemetry.activity import log_activity, monitor_with_activity

from common.constants import (
    COMPONENT_NAME,
    DEFAULT_PROPORTION,
    DEFAULT_SEED,
    TelemetryConstants,
    SUPPORTED_FILE_FORMATS,
)

logger = get_logger_app(
    "azureml.acft.contrib.hf.nlp.entry_point.data_augment.data_augment"
)


def get_parser():
    """
    Add arguments and returns the parser. Here we add all the arguments for all the tasks.

    Those arguments that are not relevant for the input task should be ignored.
    """
    parser = argparse.ArgumentParser(
        description="Data Augmentation Argument Parser", allow_abbrev=False
    )

    # File I/O
    parser.add_argument(
        "--synthetic_data_file_path",
        default=None,
        type=str,
        help="Input Synthetic data file path",
        required=True,
    )

    parser.add_argument(
        "--raw_data_file_path",
        default=None,
        type=str,
        help="Original file path",
        required=True,
    )

    parser.add_argument(
        "--proportion_percentage",
        type=float,
        required=False,
        default=DEFAULT_PROPORTION,
        help="A float value specifying the mix of original to synthetic data ranges between 0 to 1.0",
    )

    parser.add_argument(
        "--seed",
        type=int,
        required=False,
        default=DEFAULT_SEED,
        help="Seed Value for sampling the raw original data",
    )

    parser.add_argument(
        "--generated_data_file_path",
        type=Path,
        default=None,
        help="file to save the generated data",
    )

    return parser


class DataAugmentator:
    """Dataclass for Augmentation of the original Data."""

    def __init__(self, args: Namespace) -> None:
        """Initialise Data Augmentator.

        Args:
            args (Namespace): Inputs flags to validate.
        """
        self.synthetic_data_file_path = args.synthetic_data_file_path
        self.raw_data_file_path = args.raw_data_file_path
        self.proportion_percentage = args.proportion_percentage
        self.generated_data_file_path = args.generated_data_file_path
        self.seed = args.seed
        self._data_augment()

    def _data_augment(self):
        # validate file exists
        self._validate_file_exists()
        logger.info("Both File Exists")
        # validate file formats
        self._validate_file_paths_with_supported_formats()
        logger.info("Supported File format validation successful.")
        self._validate_proportion_percentage()
        logger.info("Proportion Percentage validation successful.")
        self._validate_seed()
        logger.info("Seed validation successful.")
        self._compare_jsonl_formats()
        logger.info("Supported Data format successful.")
        self._process_data()
        logger.info("Data Augmentation successful.")

    def _validate_file_paths_with_supported_formats(self):
        """Check if the file path is in the list of supported formats."""
        file_paths = [self.synthetic_data_file_path, self.raw_data_file_path]
        for file_path in file_paths:
            if file_path:
                file_suffix = Path(file_path).suffix.lower()
                file_ext = file_suffix.split("?")[0]
            if file_ext and file_ext not in SUPPORTED_FILE_FORMATS:
                raise ACFTValidationException._with_error(
                    AzureMLError.create(
                        ACFTUserError,
                        pii_safe_message=(
                            f"{file_path} is not in list of supported file formats. "
                            f"Supported file formats: {SUPPORTED_FILE_FORMATS}"
                        ),
                    )
                )

    def _validate_file_exists(self):
        """Check if the file exist."""
        file_paths = [self.synthetic_data_file_path, self.raw_data_file_path]
        for file_path in file_paths:
            if file_path:
                file = Path(file_path)
                if not file.exists():
                    raise ACFTValidationException._with_error(
                        AzureMLError.create(
                            ACFTUserError,
                            pii_safe_message=(f"File {file_path} does not exist."),
                        )
                    )

    def _validate_proportion_percentage(self):
        """Validate if proportion_percentage is well within limits."""
        if self.proportion_percentage < 0 or self.proportion_percentage > 1:
            raise ACFTValidationException._with_error(
                AzureMLError.create(
                    ACFTUserError,
                    pii_safe_message=(
                        "Invalid proportion_percentage. ",
                        f"Value should 0<=val<=1, but is {proportion_percentage}",
                    ),
                )
            )

    def _validate_seed(self):
        """Validate if seed is well within limits."""
        if not (0 <= self.seed <= (2**32 - 1)):
            raise ACFTValidationException._with_error(
                AzureMLError.create(
                    ACFTUserError,
                    pii_safe_message=(
                        "Invalid seed. ",
                        f"Value should 0<=seed<=2^32-1, but is {seed}",
                    ),
                )
            )

    def _read_jsonl_file(self, file_path):
        data = []
        with open(file_path, "r") as file:
            for line in file:
                data.append(json.loads(line))
        return data

    def _write_jsonl_file(self, file_path, data):
        with open(file_path, "w") as file:
            for item in data:
                file.write(json.dumps(item) + "\n")

    def _extract_structure(self, json_obj, parent_key=""):
        structure = {}
        for k, v in json_obj.items():
            full_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                structure[full_key] = "dict"
                structure.update(self._extract_structure(v, full_key))
            elif isinstance(v, list):
                structure[full_key] = "list"
                if v and isinstance(v[0], dict):
                    structure.update(self._extract_structure(v[0], full_key))
            else:
                structure[full_key] = type(v).__name__
        return structure

    def _compare_structures(self, structure1, structure2):
        return structure1 == structure2

    def _get_structure_from_jsonl(self, file_path):
        data = self._read_jsonl_file(file_path)
        if data:
            return self._extract_structure(data[0])
        return {}

    def _compare_jsonl_formats(self):
        structure_file1 = self._get_structure_from_jsonl(self.synthetic_data_file_path)
        structure_file2 = self._get_structure_from_jsonl(self.raw_data_file_path)
        if not self._compare_structures(structure_file1, structure_file2):
            raise ACFTValidationException._with_error(
                AzureMLError.create(
                    ACFTUserError,
                    pii_safe_message=("The JSONL files have different formats.",),
                )
            )

    def _process_data(self):
        """Process the data augmentation."""
        random.seed(self.seed)
        data1 = self._read_jsonl_file(self.synthetic_data_file_path)
        data2 = self._read_jsonl_file(self.raw_data_file_path)
        sample_size = max(1, int(self.proportion_percentage * len(data2)))
        sample_data = random.sample(data2, sample_size)
        combined_data = data1 + sample_data
        random.shuffle(combined_data)
        self._write_jsonl_file(self.generated_data_file_path, combined_data)


@swallow_all_exceptions(time_delay=5)
def main():
    """Parse args and import model."""
    parser = get_parser()
    args, _ = parser.parse_known_args()

    set_logging_parameters(
        task_type="DataAugmentation",
        acft_custom_dimensions={
            LoggingLiterals.PROJECT_NAME: PROJECT_NAME,
            LoggingLiterals.PROJECT_VERSION_NUMBER: VERSION,
            LoggingLiterals.COMPONENT_NAME: COMPONENT_NAME,
        },
        azureml_pkg_denylist_logging_patterns=LOGS_TO_BE_FILTERED_IN_APPINSIGHTS,
        log_level=logging.INFO,
    )
    with log_activity(logger=logger, activity_name=TelemetryConstants.DATA_AUGMENTATOR):
        DataAugmentator(args)


if __name__ == "__main__":
    main()

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""File containing function for FTaaS data import component."""

import argparse
from argparse import Namespace
from typing import List

from pathlib import Path
import shutil

from azureml.acft.common_components import get_logger_app, set_logging_parameters, LoggingLiterals
from azureml.acft.common_components.utils.error_handling.exceptions import ACFTValidationException
from azureml.acft.common_components.utils.error_handling.error_definitions import ACFTUserError
from azureml.acft.common_components.utils.error_handling.swallow_all_exceptions_decorator import (
    swallow_all_exceptions,
)
from azureml._common._error_definition.azureml_error import AzureMLError

from azureml.acft.contrib.hf import VERSION, PROJECT_NAME
from azureml.acft.contrib.hf.nlp.constants.constants import LOGS_TO_BE_FILTERED_IN_APPINSIGHTS


logger = get_logger_app("azureml.acft.contrib.hf.scripts.components.scripts.data_import.data_import")


SUPPORTED_FILE_FORMATS = [".jsonl"]
COMPONENT_NAME = "ACFT-Data_Import"
TRAIN_FILE_NAME = "train_input.jsonl"
VALIDATION_FILE_NAME = "validation_input.jsonl"


def get_parser():
    """
    Add arguments and returns the parser. Here we add all the arguments for all the tasks.

    Those arguments that are not relevant for the input task should be ignored.
    """
    parser = argparse.ArgumentParser(description="Model selector for hugging face models", allow_abbrev=False)

    parser.add_argument(
        "--task_name",
        type=str,
        help="Finetuning task name",
    )

    parser.add_argument(
        "--train_file_path",
        type=str,
        help="Input train file path",
    )

    parser.add_argument(
        "--validation_file_path",
        default=None,
        type=str,
        help="Input validation file path",
    )

    # Task settings
    parser.add_argument(
        "--output_dataset",
        type=Path,
        default=None,
        help="Folder to save the training data",
    )

    return parser


def _validate_file_paths_with_supported_formats(file_paths: List[str]):
    """Check if the file path is in the list of supported formats."""
    global SUPPORTED_FILE_FORMATS

    for file_path in file_paths:
        if not Path(file_path).suffix.lower() in SUPPORTED_FILE_FORMATS:
            raise ACFTValidationException._with_error(
                AzureMLError.create(
                    ACFTUserError,
                    pii_safe_message=(
                        f"{file_path} is not in list of supported file formats. "
                        f"Supported file formats: {SUPPORTED_FILE_FORMATS}"
                    )
                )
            )


def data_import(args: Namespace):
    """Copy the user data to output dir."""
    # create the directory
    Path(args.output_dataset).mkdir(exist_ok=True, parents=True)

    # validate file formats
    _validate_file_paths_with_supported_formats([args.train_file_path, args.validation_file_path])

    # copy files
    shutil.copyfile(args.train_file_path, args.output_dataset / TRAIN_FILE_NAME)
    if args.validation_file_path is not None:
        shutil.copyfile(args.validation_file_path, args.output_dataset / VALIDATION_FILE_NAME)


@swallow_all_exceptions(time_delay=5)
def main():
    """Parse args and import model."""
    # args
    parser = get_parser()
    args, _ = parser.parse_known_args()
    logger.info(args)

    set_logging_parameters(
        task_type=args.task_name,
        acft_custom_dimensions={
            LoggingLiterals.PROJECT_NAME: PROJECT_NAME,
            LoggingLiterals.PROJECT_VERSION_NUMBER: VERSION,
            LoggingLiterals.COMPONENT_NAME: COMPONENT_NAME
        },
        azureml_pkg_denylist_logging_patterns=LOGS_TO_BE_FILTERED_IN_APPINSIGHTS,
    )

    data_import(args)


if __name__ == "__main__":
    main()

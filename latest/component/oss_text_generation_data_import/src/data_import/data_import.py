# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""File containing function for FTaaS data import component."""

import argparse
from argparse import Namespace

from pathlib import Path
import shutil

from azureml.acft.common_components import get_logger_app, set_logging_parameters, LoggingLiterals
from azureml.acft.common_components.utils.error_handling.swallow_all_exceptions_decorator import (
    swallow_all_exceptions,
)

from azureml.acft.contrib.hf import VERSION, PROJECT_NAME
from azureml.acft.contrib.hf.nlp.constants.constants import LOGS_TO_BE_FILTERED_IN_APPINSIGHTS


logger = get_logger_app("azureml.acft.contrib.hf.scripts.components.scripts.data_import.data_import")


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


def data_import(args: Namespace):
    """Copy the user data to output dir."""
    # create the directory
    Path(args.output_dataset).mkdir(exist_ok=True)

    shutil.copyfile(args.train_file_path, args.output_dataset / TRAIN_FILE_NAME)
    # TODO Keep validation file optional
    shutil.copyfile(args.validation_file_path, args.output_dataset / VALIDATION_FILE_NAME)


@swallow_all_exceptions(time_delay=60)
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

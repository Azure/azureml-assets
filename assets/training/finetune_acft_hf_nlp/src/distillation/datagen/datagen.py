# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Read the args from preprocess component."""

import json
import argparse
from pathlib import Path
from argparse import Namespace
from typing import Any
from functools import partial
import logging

from transformers.trainer_utils import get_last_checkpoint

from azureml.acft.contrib.hf.nlp.task_factory import get_task_runner
from azureml.acft.contrib.hf.nlp.constants.constants import (
    SaveFileConstants,
    Tasks,
    HfModelTypes,
    LOGS_TO_BE_FILTERED_IN_APPINSIGHTS,
)
from azureml.acft.contrib.hf.nlp.utils.data_utils import copy_and_overwrite, clean_column_name
from azureml.acft.contrib.hf.nlp.nlp_auto.config import AzuremlAutoConfig
from azureml.acft.contrib.hf.nlp.tasks.translation.preprocess.preprocess_for_finetune import T5_CODE2LANG_MAP

from azureml.acft.common_components.utils.error_handling.exceptions import ACFTValidationException
from azureml.acft.common_components.utils.error_handling.error_definitions import (
    PathNotFound,
    ACFTUserError,
)
from azureml.acft.common_components.utils.error_handling.swallow_all_exceptions_decorator import (
    swallow_all_exceptions,
)
from azureml.acft.common_components import get_logger_app, set_logging_parameters, LoggingLiterals
from azureml.acft.contrib.hf import VERSION, PROJECT_NAME
from azureml._common._error_definition.azureml_error import AzureMLError  # type: ignore


logger = get_logger_app("azureml.acft.contrib.hf.scripts.src.preprocess.preprocess")

COMPONENT_NAME = "ACFT-Preprocess"


def str2bool(arg):
    """Convert string to bool."""
    arg = arg.lower()
    if arg in ["true", '1']:
        return True
    elif arg in ["false", '0']:
        return False
    else:
        raise ValueError(f"Invalid argument {arg} to while converting string to boolean")


def default_missing_path(arg_str: str, default: Any = None):
    """If path of the arg is missing, reset the value to default. Use this type with paths."""
    if isinstance(arg_str, str) and not Path(arg_str).is_file():
        return default
    return arg_str


def get_parser():
    """
    Add arguments and returns the parser. Here we add all the arguments for all the tasks.

    Those arguments that are not relevant for the input task should be ignored.
    """
    parser = argparse.ArgumentParser(description="Model Preprocessing", allow_abbrev=False)

    # NOTE that the default is present in both :param `type` and `default`. In case of change, we need to update
    # in both places
    parser.add_argument(
        "--source_file_path",
        type=partial(default_missing_path, default=None),
        required=False,
        default=None,
        help="Train data path",
    )
    # NOTE that the default is present in both :param `type` and `default`. In case of change, we need to update
    # in both places
    parser.add_argument(
        "--source_mltable_path",
        type=str,
        required=False,
        default=None,
        help="train mltable dataset_path folder",
    )

    parser.add_argument(
        "--output_file",
        default="preprocess_output",
        type=str,
        help="file to store generated input data",
    )

    # Add openai inputs

    return parser


def distillation_datagen(parsed_args: Namespace, unparsed_args: list):
    """Datagen."""
    # TODO: use GPT4 or any teacher model to generate response, and write to a jsonl as input to FT
    # example GPT 4 call:
    import os
    import openai

    openai.api_type = "azure"
    openai.api_base = "https://daholsteauseaoai.openai.azure.com/"
    openai.api_version = "2023-07-01-preview"
    openai.api_key = os.getenv("OPENAI_API_KEY") # 6e236f9f4ee84a54958be3d4bc05d88f

    message_text = [{"role":"system","content":"You are an AI assistant that helps people find information."},{"role":"user","content":"what is 2+2?"},{"role":"assistant","content":"2 + 2 equals 4."}]

    completion = openai.ChatCompletion.create(
    engine="gpt4",
    messages = message_text,
    temperature=0.7,
    max_tokens=800,
    top_p=0.95,
    frequency_penalty=0,
    presence_penalty=0,
    stop=None
    )


@swallow_all_exceptions(time_delay=60)
def main():
    """Parse args and pre process."""
    parser = get_parser()
    # unknown args are the command line strings that could not be parsed by the argparser
    parsed_args, unparsed_args = parser.parse_known_args()
    logger.info(f"Component Args: {parsed_args}")

    set_logging_parameters(
        task_type=parsed_args.task_name,
        acft_custom_dimensions={
            LoggingLiterals.PROJECT_NAME: PROJECT_NAME,
            LoggingLiterals.PROJECT_VERSION_NUMBER: VERSION,
            LoggingLiterals.COMPONENT_NAME: COMPONENT_NAME
        },
        azureml_pkg_denylist_logging_patterns=LOGS_TO_BE_FILTERED_IN_APPINSIGHTS,
        log_level=logging.INFO,
    )

    distillation_datagen(parsed_args, unparsed_args)


if __name__ == "__main__":
    main()

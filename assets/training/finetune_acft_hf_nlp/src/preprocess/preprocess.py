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

from azureml.acft.common_components.utils.error_handling.exceptions import ACFTValidationException, ACFTSystemException
from azureml.acft.common_components.utils.error_handling.error_definitions import (
    PathNotFound,
    ACFTSystemError,
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
        "--train_file_path",
        type=partial(default_missing_path, default=None),
        required=False,
        default=None,
        help="Train data path",
    )
    # NOTE that the default is present in both :param `type` and `default`. In case of change, we need to update
    # in both places
    parser.add_argument(
        "--validation_file_path",
        type=partial(default_missing_path, default=None),
        required=False,
        default=None,
        help="Validation data path",
    )
    # NOTE that the default is present in both :param `type` and `default`. In case of change, we need to update
    # in both places
    parser.add_argument(
        "--test_file_path",
        type=partial(default_missing_path, default=None),
        required=False,
        default=None,
        help="Test data path",
    )
    parser.add_argument(
        "--train_mltable_path",
        type=str,
        required=False,
        default=None,
        help="train mltable dataset_path folder",
    )
    parser.add_argument(
        "--validation_mltable_path",
        type=str,
        required=False,
        default=None,
        help="valid mltable dataset_path folder",
    )
    parser.add_argument(
        "--test_mltable_path",
        type=str,
        required=False,
        default=None,
        help="valid mltable dataset_path folder",
    )
    # `test_data` is used as pass through and will be consumed directly by the model evaluation component
    # Instead of deleting the entire code related to test processing; introducing a new flag that enables/disables
    # processing test data
    # Enabling this flag will skip data validation and data encoding for the test data
    parser.add_argument(
        "--skip_test_data_processing",
        type=str2bool,
        required=False,
        default="true",
        help="If enabled, the processing for test data will be skipped",
    )

    parser.add_argument(
        "--output_dir",
        default="preprocess_output",
        type=str,
        help="folder to store preprocessed input data",
    )

    # Task settings
    parser.add_argument(
        "--model_selector_output",
        default=None,
        type=str,
        help=(
            "output folder of model selector containing model configs, tokenizer, checkpoints in case of model_id."
            "If huggingface_id is selected, the model download happens dynamically on the fly"
        ),
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="SingleLabelClassification",
        help="Task Name",
    )

    # NLP settings
    parser.add_argument(
        "--enable_long_range_text",
        type=bool,
        default=True,
        help=(
            "User option to apply heuristic for finding optimal max_seq_length value, "
            "reserved only for multiclass classification."
        )
    )

    return parser


def pre_process(parsed_args: Namespace, unparsed_args: list):
    """Pre process data."""
    Path(parsed_args.output_dir).mkdir(exist_ok=True, parents=True)

    # Model Selector Component ---> Preprocessor Component
    model_selector_args_path = Path(parsed_args.model_selector_output, SaveFileConstants.MODEL_SELECTOR_ARGS_SAVE_PATH)
    if not model_selector_args_path.exists():
        raise ACFTValidationException._with_error(AzureMLError.create(PathNotFound, path=model_selector_args_path))

    with open(model_selector_args_path, "r") as rptr:
        model_selector_args = json.load(rptr)
        parsed_args.model_asset_id = model_selector_args.get("model_asset_id")
        parsed_args.model_name = model_selector_args.get("model_name")
        model_name_or_path = Path(parsed_args.model_selector_output, parsed_args.model_name)
        # Transformers lib searches for tokenizer files locally only if the folder path is same as model's name
        if model_name_or_path.is_dir():
            last_checkpoint = get_last_checkpoint(model_name_or_path)
            if last_checkpoint:
                model_name_or_path = last_checkpoint
            logger.info(f"Copying content from {model_name_or_path} to {parsed_args.model_name}")
            copy_and_overwrite(str(model_name_or_path), parsed_args.model_name)
        parsed_args.model_name_or_path = parsed_args.model_name

    # read FT config
    ft_config_path = Path(parsed_args.model_selector_output, SaveFileConstants.ACFT_CONFIG_SAVE_PATH)
    if ft_config_path.is_file():
        with open(ft_config_path, "r") as rptr:
            ft_config = json.load(rptr)
            setattr(parsed_args, "finetune_config", ft_config)
            logger.info("Added finetune config to `component_args`")
    else:
        logger.info(f"{SaveFileConstants.ACFT_CONFIG_SAVE_PATH} does not exist")
        setattr(parsed_args, "finetune_config", {})

    # additional logging
    logger.info(f"Model name: {getattr(parsed_args, 'model_name', None)}")
    logger.info(f"Task name: {getattr(parsed_args, 'task_name', None)}")
    logger.info(f"Model asset id: {getattr(parsed_args, 'model_asset_id', None)}")

    if getattr(parsed_args, "task_name", None) == Tasks.TRANSLATION and \
            getattr(parsed_args, "model_name", None) is not None:
        model_type, src_lang, tgt_lang = None, None, None
        try:
            src_lang_idx = unparsed_args.index("--source_lang")
            src_lang = clean_column_name(unparsed_args[src_lang_idx + 1])
            tgt_lang_idx = unparsed_args.index("--target_lang")
            tgt_lang = clean_column_name(unparsed_args[tgt_lang_idx + 1])
            # fetching model_name as path is already updated above to model_name
            model_type = AzuremlAutoConfig.get_model_type(hf_model_name_or_path=parsed_args.model_name)
        except Exception as e:
            logger.info(f"Unable to parse languages, continuing preprocess! - {e}")

        if model_type == HfModelTypes.T5 and \
                (src_lang not in T5_CODE2LANG_MAP or tgt_lang not in T5_CODE2LANG_MAP):
            raise ACFTValidationException._with_error(
                AzureMLError.create(ACFTUserError, pii_safe_message=(
                    "Either source or target language is not supported for T5. Supported languages are "
                    f"{list(T5_CODE2LANG_MAP.keys())}"
                ))
            )

    # update dataset paths
    parsed_args.train_data_path = parsed_args.train_file_path
    parsed_args.validation_data_path = parsed_args.validation_file_path
    parsed_args.test_data_path = parsed_args.test_file_path
    if parsed_args.train_mltable_path:
        parsed_args.train_data_path = getattr(parsed_args, "train_mltable_path")
    if parsed_args.validation_mltable_path:
        parsed_args.validation_data_path = getattr(parsed_args, "validation_mltable_path")
    if parsed_args.test_mltable_path:
        parsed_args.test_data_path = getattr(parsed_args, "test_mltable_path")

    # raise errors and warnings
    if not parsed_args.train_data_path:
        raise ACFTSystemException._with_error(
            AzureMLError.create(ACFTSystemError, pii_safe_message=(
                "train_file_path or train_mltable_path need to be passed"
            ))
        )
    if parsed_args.train_file_path and parsed_args.train_mltable_path:
        logger.warning("Passed trainfile_path and train_mltable_path, considering train_mltable_path only")
    if parsed_args.validation_file_path and parsed_args.validation_mltable_path:
        logger.warning("Passed validation_file_path and validation_mltable_path, "
                       "considering validation_mltable_path only")
    if parsed_args.test_file_path and parsed_args.test_mltable_path:
        logger.warning("Passed test_file_path and test_mltable_path, considering test_mltable_path only")

    # handeling data splits
    if not parsed_args.validation_data_path and not parsed_args.test_data_path:
        # user passed only train data
        # enforce validation_slice_percent and test_slice_percent
        # the default slice ratio is train:validation:test are 80:10:10
        parsed_args.train_slice_percent = 80
        parsed_args.validation_slice_percent = 10
        parsed_args.test_slice_percent = 10
        parsed_args.validation_data_path = parsed_args.train_data_path
        parsed_args.test_data_path = parsed_args.train_data_path
    elif not parsed_args.validation_data_path:
        # user passed train and test files
        # enforce validation_slice_percent
        # the default slice ratio for train:validation is 80:20
        parsed_args.train_slice_percent = 80
        parsed_args.validation_slice_percent = 20
        parsed_args.test_slice_percent = 0
        parsed_args.validation_data_path = parsed_args.train_data_path
    elif not parsed_args.test_data_path:
        # user passed train and validation files
        # enforce test_slice_percent
        # the default slice ratio for train:test are 80:20
        parsed_args.train_slice_percent = 80
        parsed_args.validation_slice_percent = 0
        parsed_args.test_slice_percent = 20
        parsed_args.test_data_path = parsed_args.train_data_path
    else:
        # user passed train, validation and test files
        parsed_args.train_slice_percent = 0
        parsed_args.validation_slice_percent = 0
        parsed_args.test_slice_percent = 0
    logger.info(f"train slice percent = {parsed_args.train_slice_percent}")
    logger.info(f"validation slice percent of train data = {parsed_args.validation_slice_percent}")
    logger.info(f"test slice percent of train data = {parsed_args.test_slice_percent}")
    # preparing `slice` string for dataset
    parsed_args.train_slice = f"train[:{parsed_args.train_slice_percent}%]" \
        if parsed_args.train_slice_percent != 0 else "train"
    parsed_args.validation_slice = (
        f"train[{parsed_args.train_slice_percent}%:"
        f"{(parsed_args.train_slice_percent + parsed_args.validation_slice_percent)}%]"
    ) if parsed_args.validation_slice_percent != 0 else "train"
    parsed_args.test_slice = f"train[{(-1*parsed_args.test_slice_percent)}%:]" \
        if parsed_args.test_slice_percent != 0 else "train"

    # Preprocessing component has `unparsed args` which will be parsed and returned after this method
    hf_task_runner = get_task_runner(task_name=parsed_args.task_name)()
    hf_task_runner.run_preprocess_for_finetune(parsed_args, unparsed_args)  # type: ignore


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

    pre_process(parsed_args, unparsed_args)


if __name__ == "__main__":
    main()

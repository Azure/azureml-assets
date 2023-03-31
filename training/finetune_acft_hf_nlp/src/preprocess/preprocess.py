# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Read the args from preprocess component."""

import json
import argparse
from pathlib import Path
from argparse import Namespace


from azureml.acft.contrib.hf.nlp.task_factory import get_task_runner
from azureml.acft.contrib.hf.nlp.constants.constants import SaveFileConstants
from azureml.acft.contrib.hf.nlp.utils.data_utils import copy_and_overwrite

from azureml.acft.accelerator.utils.logging_utils import get_logger_app
from azureml.acft.accelerator.utils.error_handling.exceptions import ValidationException, LLMException
from azureml.acft.accelerator.utils.error_handling.error_definitions import PathNotFound, LLMInternalError
from azureml.acft.accelerator.utils.decorators import swallow_all_exceptions
from azureml._common._error_definition.azureml_error import AzureMLError  # type: ignore


logger = get_logger_app()


def str2bool(arg):
    """Convert string to bool."""
    arg = arg.lower()
    if arg in ["true", '1']:
        return True
    elif arg in ["false", '0']:
        return False
    else:
        raise ValueError(f"Invalid argument {arg} to while converting string to boolean")


def get_parser():
    """
    Add arguments and returns the parser. Here we add all the arguments for all the tasks.

    Those arguments that are not relevant for the input task should be ignored.
    """
    parser = argparse.ArgumentParser(description="Model Preprocessing", allow_abbrev=False)

    parser.add_argument(
        "--train_file_path",
        type=str,
        required=False,
        default=None,
        help="Train data path",
    )
    parser.add_argument(
        "--validation_file_path",
        type=str,
        required=False,
        default=None,
        help="Validation data path",
    )
    parser.add_argument(
        "--test_file_path",
        type=str,
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
        raise ValidationException._with_error(AzureMLError.create(PathNotFound, path=model_selector_args_path))

    with open(model_selector_args_path, "r") as rptr:
        model_selector_args = json.load(rptr)
        parsed_args.model_name = model_selector_args.get("model_name")
        model_name_or_path = Path(parsed_args.model_selector_output, parsed_args.model_name)
        # Transformers lib searches for tokenizer files locally only if the folder path is same as model's name
        if model_name_or_path.is_dir():
            copy_and_overwrite(str(model_name_or_path), parsed_args.model_name)
        parsed_args.model_name_or_path = parsed_args.model_name

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
        raise LLMException._with_error(
            AzureMLError.create(LLMInternalError, error=(
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


@swallow_all_exceptions(logger)
def main():
    """Parse args and pre process."""
    parser = get_parser()
    # unknown args are the command line strings that could not be parsed by the argparser
    parsed_args, unparsed_args = parser.parse_known_args()
    logger.info(f"Component Args: {parsed_args}")

    pre_process(parsed_args, unparsed_args)


if __name__ == "__main__":
    main()

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
File containing function for preprocess component.
"""

import os
import json
import argparse

from azureml.train.finetune.core.drivers.preprocess import validate_and_preprocess

# from azureml.gllm.model_selector.definitions import DecodeUserTask, ProblemType
# from azureml.gllm.model_selector.argument_parser import parse_task_args
from azureml.train.finetune.core.constants import task_definitions
from azureml.train.finetune.core.constants.constants import SaveFileConstants

from azureml.train.finetune.core.utils.logging_utils import get_logger_app

from azureml.train.finetune.core.utils.error_handling.exceptions import (
    ValidationException,
    ArgumentException,
)
from azureml.train.finetune.core.utils.error_handling.error_definitions import (
    TaskNotSupported,
    EmptyLabelSeparator,
)
from azureml.train.finetune.core.utils.error_handling.error_definitions import PathNotFound
from azureml._common._error_definition.azureml_error import AzureMLError  # type: ignore

# from utils.decorators import swallow_all_exceptions


logger = get_logger_app()


def get_task_parser(task_metadata):
    """Parse all arguments."""
    parser = argparse.ArgumentParser(task_metadata.hf_task_name, allow_abbrev=False)
    for item in task_metadata.arg_parse_data:
        arg_name = item["dest"]
        key = f"--{arg_name}"
        parser.add_argument(key, **item)

    return parser


def get_common_parser():
    """
    Adds arguments and returns the parser. Here we add all the arguments for all the tasks.
    Those arguments that are not relevant for the input task should be ignored
    """
    parser = argparse.ArgumentParser(description="Preprocessing for hugging face models", allow_abbrev=False)

    # NOTE Only train and validation datasets are used for finetuning
    # This is because test is not used during finetuning and also
    # the test might not contain the output label and requires different handling
    parser.add_argument(
        "--train_file_path",
        type=str,
        required=True,
        help="train file name in dataset_path folder",
    )
    parser.add_argument(
        "--valid_file_path",
        type=str,
        required=True,
        help="valid file name in dataset_path folder",
    )
    parser.add_argument(
        "--output_dir",
        default="preprocess_output",
        type=str,
        help="folder to store preprocessed outputs of input data",
    )

    # Tokenizer settings
    # NOTE Add a note for the user
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=-1,
        help=(
            "Max tokens of single example, set the value to -1 to use the default value."
            "Default value will be max seq length of pretrained model tokenizer"
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        type=str,
        default="true",
        help=(
            "If true, all samples get padded to `max_seq_length`."
            "If false, will pad the samples dynamically when batching to the maximum length in the batch."
        ),
    )

    # Task settings
    parser.add_argument(
        "--model_path",
        default=None,
        type=str,
        help=("output folder of model selector containing model configs, tokenizer, checkpoints."),
    )
    # parser.add_argument("--label_names", type=str, default="None", help="comma separated label names")
    parser.add_argument(
        "--task_name",
        type=str,
        default="SingleLabelClassification",
        help="NLP Task Name",
    )

    return parser


if __name__ == "__main__":
    # common args
    common_parser = get_common_parser()
    common_args, _ = common_parser.parse_known_args()

    model_selector_args_path = os.path.join(common_args.model_path, SaveFileConstants.ModelSelectorArgsSavePath)

    if not os.path.exists(model_selector_args_path):
        raise ValidationException._with_error(AzureMLError.create(PathNotFound, path=model_selector_args_path))

    with open(model_selector_args_path, "r") as rptr:
        model_selector_args = json.load(rptr)
        common_args.model_name_or_path = model_selector_args.get("model_name_or_path")

    # decode user task name to HF task name
    common_args.user_task_name = common_args.task_name
    user_task = getattr(task_definitions, common_args.user_task_name, None)

    if user_task is None:
        raise ValidationException._with_error(
            AzureMLError.create(TaskNotSupported, TaskName=common_args.user_task_name)
        )

    task_metadata = user_task()
    common_args.hf_task_name = task_metadata.hf_task_name
    common_args.hf_problem_type = task_metadata.hf_problem_type

    # add task related arguments
    task_parser = get_task_parser(task_metadata)
    task_args, _ = task_parser.parse_known_args()

    # combine common args and task related args
    args = argparse.Namespace(**vars(common_args), **vars(task_args))

    # Check if the label_separator is not empty
    if (getattr(args, "label_separator", None) is not None) and len(args.label_separator) == 0:
        raise ArgumentException._with_error(AzureMLError.create(EmptyLabelSeparator))

    # Converting string to bool
    if isinstance(args.pad_to_max_length, str):
        args.pad_to_max_length = args.pad_to_max_length.lower() == "true"

    padding_strategy = (
        "padding to tokenizer max length" if args.pad_to_max_length else "padding to input sequence length"
    )
    logger.info(f"Padding strategy: {padding_strategy} ")

    if getattr(args, "label_all_tokens", None):
        if isinstance(args.label_all_tokens, str):
            args.label_all_tokens = args.label_all_tokens.lower() == "true"

        tokenization_preprocessing_strategy = (
            "label all tokens" if args.label_all_tokens else "label only the first token"
        )
        logger.info(f"Preprocessing strategy for TokenClassification: {tokenization_preprocessing_strategy}")

    # setting the train and validation data paths
    # TODO: Clean this later
    args.train_data_path = args.train_file_path
    args.valid_data_path = args.valid_file_path

    # # construct label map
    # args.label_map = get_label_map(args.problem_type)
    # logger.info(args)

    # update the task info
    args.dataset_target_key = getattr(args, task_metadata.dataset_target_key)
    decode_dataset_columns = []
    decode_datast_columns_dtypes = []
    for idx, var in enumerate(task_metadata.dataset_columns):
        decoded_arg = getattr(args, var)  # This will translate to, for example, prompt = args.sentence1_key
        if decoded_arg is not None:
            decode_dataset_columns.append(decoded_arg)
            decode_datast_columns_dtypes.append(task_metadata.dataset_columns_dtypes[idx])
    args.keep_columns = decode_dataset_columns
    args.keep_columns_dtypes = decode_datast_columns_dtypes
    logger.info(args)

    validate_and_preprocess(args)

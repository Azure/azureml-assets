# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Read the args from preprocess component"""

import argparse
import json
from pathlib import Path
from argparse import Namespace
from typing import List

from azureml.acft.multimodal.components import PROJECT_NAME, VERSION
from azureml.acft.multimodal.components.constants.constants import SaveFileConstants, ProblemType, Tasks
from azureml.acft.multimodal.components.task_factory import get_task_runner
from azureml.acft.contrib.hf.nlp.utils.data_utils import copy_and_overwrite

from azureml.acft.accelerator.utils.error_handling.exceptions import ValidationException
from azureml.acft.accelerator.utils.error_handling.error_definitions import PathNotFound
from azureml._common._error_definition.azureml_error import AzureMLError  # type: ignore

from azureml.acft.common_components import get_logger_app, set_logging_parameters, LoggingLiterals


logger = get_logger_app("azureml.acft.multimodal.components.scripts.components.preprocess.preprocess")


def get_parser():
    """
    Adds arguments and returns the parser.
    """
    parser = argparse.ArgumentParser(description="Tabular Preprocessing", allow_abbrev=False)

    parser.add_argument(
        "--problem_type",
        default=ProblemType.SINGLE_LABEL_CLASSIFICATION,
        type=str,
        help="Whether its single label or multilabel classification",
    )
    parser.add_argument(
        "--train_mltable_path",
        type=str,
        required=True,
        help="Train mltable path",
    )
    parser.add_argument(
        "--validation_mltable_path",
        type=str,
        required=True,
        help="Validation mltable path",
    )
    parser.add_argument(
        "--test_mltable_path",
        type=str,
        required=False,
        help="Test mltable path",
    )

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
        "--output_dir",
        default="preprocess_output",
        type=str,
        help="folder to store model selector output and metadata for preprocessed input data",
    )

    # Tabular preprocessor settings
    parser.add_argument(
        "--label_column",
        type=str,
        required=True,
        help="Target label column name",
    )
    parser.add_argument(
        "--image_column",
        type=str,
        required=True,
        help="Image column name",
    )
    parser.add_argument(
        "--drop_columns",
        type=str,
        default="",
        help="Columns to ignore in the input data. Should be a comma-separated list of column names. "
        "Example: 'column_1,column_2'",
    )
    parser.add_argument(
        "--numerical_columns_overrides",
        type=str,
        default="",
        help="Columns to treat as numerical in the input data. This setting would override the column types detected "
        "from automatic column purpose detection. Should be a comma-separated list of column names. "
        "Example: 'column_1,column_2'",
    )
    parser.add_argument(
        "--categorical_columns_overrides",
        type=str,
        default="",
        help="Columns to treat as categorical in the input data. This setting would override the column types "
        "detected from automatic column purpose detection. Should be a comma-separated list of column names. "
        "Example: 'column_1,column_2'",
    )
    parser.add_argument(
        "--text_columns_overrides",
        type=str,
        default="",
        help="Columns to treat as text in the input data. This setting would override the column types detected "
        "from automatic column purpose detection. Should be a comma-separated list of column names. "
        "Example: 'column_1,column_2'",
    )

    return parser


def pre_process(parsed_args: Namespace, unparsed_args: List[str]):
    """
    main function handling tabular data preprocess
    """
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
    if parsed_args.train_mltable_path and parsed_args.validation_mltable_path:
        parsed_args.train_data_path = getattr(parsed_args, "train_mltable_path")
        parsed_args.validation_data_path = getattr(parsed_args, "validation_mltable_path")
        delattr(parsed_args, "train_mltable_path")
        delattr(parsed_args, "validation_mltable_path")
        if parsed_args.test_mltable_path:
            parsed_args.test_data_path = getattr(parsed_args, "test_mltable_path")
            delattr(parsed_args, "test_mltable_path")
        else:
            parsed_args.test_data_path = None

    parsed_args.enable_long_range_text = True
    if parsed_args.problem_type == ProblemType.SINGLE_LABEL_CLASSIFICATION:
        parsed_args.task_name = Tasks.MUTIMODAL_CLASSIFICATION
    else:
        parsed_args.task_name = Tasks.MULTIMODAL_MULTILABEL_CLASSIFICATION

    task_runner = get_task_runner(task_name=parsed_args.task_name)()
    task_runner.run_preprocess_for_finetune(parsed_args, unparsed_args)


if __name__ == "__main__":
    parser = get_parser()

    # unknown args are the command line strings that could not be parsed by the argparser
    parsed_args, unparsed_args = parser.parse_known_args()

    set_logging_parameters(
        task_type=Tasks.MUTIMODAL_CLASSIFICATION,
        acft_custom_dimensions={
            LoggingLiterals.PROJECT_NAME: PROJECT_NAME,
            LoggingLiterals.PROJECT_VERSION_NUMBER: VERSION,
        },
    )

    logger.info(f"Component Args: {parsed_args}")
    logger.info(f"preprocessor Args: {unparsed_args}")

    pre_process(parsed_args, unparsed_args)

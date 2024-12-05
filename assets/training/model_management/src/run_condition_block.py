# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Run Model preprocessor module."""

import argparse
import os
import json
import shutil
from azureml.model.mgmt.config import AppName, ModelFramework
from azureml.model.mgmt.processors.transformers.config import HF_CONF
from azureml.model.mgmt.processors.preprocess import run_preprocess, check_for_py_files
from azureml.model.mgmt.processors.transformers.config import SupportedTasks as TransformersSupportedTasks
from azureml.model.mgmt.processors.pyfunc.config import SupportedTasks as PyFuncSupportedTasks
from azureml.model.mgmt.utils.exceptions import swallow_all_exceptions, UnsupportedTaskType
from azureml._common.exceptions import AzureMLException
from azureml._common._error_definition.azureml_error import AzureMLError
from azureml.model.mgmt.utils.logging_utils import custom_dimensions, get_logger
from pathlib import Path
from tempfile import TemporaryDirectory
import json


logger = get_logger(__name__)
custom_dimensions.app_name = AppName.CONVERT_MODEL_TO_MLFLOW


def _get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", type=str, required=True, help="Condition")
    parser.add_argument(
        "--input-args",
        type=str,
        required=True,
        help="Input args",
    )
    return parser


@swallow_all_exceptions(logger)
def run():
    """Run preprocess."""
    parser = _get_parser()
    args, _ = parser.parse_known_args()

    input_args = args.input_args
    condition = args.condition
    input_args = json.loads(input_args)
    result = None
    try:
        result = eval(condition, input_args)
    except Exception as e:
        logger.error(f"Error evaluating condition: {e}")
        result = False
    

if __name__ == "__main__":
    run()

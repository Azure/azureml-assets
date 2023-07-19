# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Run MLflow model Local validations module."""

import argparse
import os
import shutil
from azureml._common._error_definition import AzureMLError
from azureml._common.exceptions import AzureMLException
from pathlib import Path

from utils.config import AppName
from utils.exceptions import swallow_all_exceptions
from utils.logging_utils import custom_dimensions, get_logger
from utils.common_utils import run_command
from utils.exceptions import (
    CondaEnvCreationError,
    CondaFileMissingError,
    MlflowModelValidationError,
)


logger = get_logger(__name__)
custom_dimensions.app_name = AppName.MLFLOW_MODEL_LOCAL_VALIDATION

ENV_PREFIX = "/azureml-envs/inferencing"
LOCAL_VALIDATION_OUT_FILE = "output.log"
CONDA_FILE_NAME = "conda.yaml"
CREATE_CONDA_CMD = "conda env create -p {} -f {} -q"
CONDA_LIST = "conda list -p {}"
CONDA_EXEC_CMD = "conda run -p {} {}"
SCRIPT_PATH = "scripts/validate_model.py"


def _get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True, help="Model input path")
    parser.add_argument("--test-data-path", type=Path, required=False, help="Test dataset path")
    parser.add_argument("--column-rename-map", type=str, required=False, help="")
    parser.add_argument("--output-model-path", type=Path, required=True, help="Output model path")
    return parser


@swallow_all_exceptions(logger)
def run():
    """Run mlflow model local validation."""
    parser = _get_parser()
    args, _ = parser.parse_known_args()

    model_dir: Path = args.model_path
    test_data_path: Path = args.test_data_path
    col_rename_map_str: str = args.column_rename_map
    output_model_path: Path = args.output_model_path

    logger.info(f"col_rename_map_str {col_rename_map_str}")
    logger.info(f"listing model dir {os.listdir(model_dir)}")

    conda_file_path = os.path.join(model_dir, CONDA_FILE_NAME)
    if not os.path.exists(conda_file_path):
        raise AzureMLException._with_error(AzureMLError.create(CondaFileMissingError))

    # copy conda.yaml to cwd
    shutil.copy(conda_file_path, CONDA_FILE_NAME)

    # create conda env
    logger.info("Creating conda env")
    exit_code, stdout = run_command(CREATE_CONDA_CMD.format(ENV_PREFIX, CONDA_FILE_NAME))
    if exit_code != 0:
        logger.warning(f"Error in creating conda env. Error details {stdout}")
        raise AzureMLException._with_error(AzureMLError.create(CondaEnvCreationError))

    exit_code, stdout = run_command(CONDA_LIST.format(ENV_PREFIX))
    if exit_code != 0:
        logger.warning(f"Error in listing env at {ENV_PREFIX}. Error details {stdout}")
        raise AzureMLException._with_error(AzureMLError.create(CondaEnvCreationError))
    logger.info(f"pip list: \n{stdout}")

    cmd = f"python {SCRIPT_PATH} --model-path {model_dir}"
    if test_data_path:
        cmd += f" --test-data-path {test_data_path}"
    if col_rename_map_str:
        cmd += f" --column-rename-map '{col_rename_map_str}'"
    cmd += f" > {LOCAL_VALIDATION_OUT_FILE} 2>&1"

    run_script_cmd = CONDA_EXEC_CMD.format(ENV_PREFIX, cmd)
    logger.info(f"cmd string: {run_script_cmd}")

    exit_code, stdout = run_command(run_script_cmd)
    if os.path.exists(LOCAL_VALIDATION_OUT_FILE):
        # read script logs and feed to logger handler here
        with open(LOCAL_VALIDATION_OUT_FILE) as f:
            for line in f:
                logger.info(f"[SCRIPT_LOG] {line}")
    else:
        logger.warning("local validation script execution log was not captured")

    if exit_code != 0:
        logger.warning(f"Local validation failed. Error {stdout}")
        raise AzureMLException._with_error(AzureMLError.create(MlflowModelValidationError))

    # copy the model to output dir
    shutil.copytree(src=model_dir, dst=output_model_path, dirs_exist_ok=True)


if __name__ == "__main__":
    run()

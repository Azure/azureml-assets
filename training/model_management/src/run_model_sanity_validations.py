# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Run Model Sanity Validations module."""

import argparse
import logging
import sys
import shutil
import yaml
from azureml.model.mgmt.utils.common_utils import run_command
from pathlib import Path


MLFLOW_MODEL_SCORING_SCRIPT = "validations/mlflow_model_scoring_script.py"
CONDA_ENV_PREFIX = "/opt/conda/envs/inferencing"
MLMODEL_FILE_NAME = "MLmodel"
CONDA_YAML_FILE_NAME = "conda.yaml"

stdout_handler = logging.StreamHandler(stream=sys.stdout)
handlers = [stdout_handler]
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
    handlers=handlers
)
logger = logging.getLogger(__name__)


def _get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True, help="Model input path")
    parser.add_argument("--test-data-path", type=Path, required=True, help="Test dataset path")
    parser.add_argument("--column-rename-map", type=str, required=False, help="")
    parser.add_argument("--output-model-path", type=Path, required=True, help="Output model path")
    return parser


if __name__ == "__main__":
    parser = _get_parser()
    args, _ = parser.parse_known_args()

    input_model_path: Path = args.model_path
    test_data_path: Path = args.test_data_path
    col_rename_map_str: str = args.column_rename_map
    output_model_path: Path = args.output_model_path

    logger.info("##### logger.info args #####")
    for arg, value in args.__dict__.items():
        logger.info(f"{arg} => {value}")

    model_dir = f"{input_model_path}"
    mlmodel_file_path = f"{model_dir}/{MLMODEL_FILE_NAME}"
    conda_env_file_path = f"{model_dir}/{CONDA_YAML_FILE_NAME}"

    with open(mlmodel_file_path) as f:
        mlmodel_dict = yaml.safe_load(f)
        logger.info(f"mlmodel :\n{mlmodel_dict}\n")

    with open(conda_env_file_path) as f:
        conda_dict = yaml.safe_load(f)
        logger.info(f"conda :\n{conda_dict}\n")

    cp_conda_yaml = f"cp {conda_env_file_path} ./"
    conda_create_env_command = f"conda env create -p {CONDA_ENV_PREFIX} -f {CONDA_YAML_FILE_NAME} -q"

    logger.info("Creating conda env using MLFlow model conda file.")
    cmd = cp_conda_yaml + " && " + conda_create_env_command
    exit_code, stdout = run_command(cmd)
    if exit_code != 0:
        raise Exception(f"Error in creating conda env. Error {stdout}")
    logger.info(f"conda env successfully created at {CONDA_ENV_PREFIX}")

    run_model_inferencing_script = (
        f"conda run -p  {CONDA_ENV_PREFIX} python {MLFLOW_MODEL_SCORING_SCRIPT}" +
        f" --model-path {model_dir}" +
        f" --test-data-path {test_data_path}" +
        f" --column-rename-map {col_rename_map_str}"
    )

    logger.info("Loading model and testing inference")
    exit_code, stdout = run_command(run_model_inferencing_script)
    if exit_code != 0:
        raise Exception(f"Error in local validation for model. Error {stdout}")
    logger.info(f"Local validation completed:\n{stdout}")

    # copy the model to output dir
    shutil.copytree(src=input_model_path, dst=output_model_path, dirs_exist_ok=True)

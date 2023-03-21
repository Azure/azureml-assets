# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Run Model Sanity Validations module."""

import argparse
import logging
import mlflow
import sys
import json
import yaml
import pandas as pd
from azureml.model.mgmt.utils.common_utils import run_command
from pathlib import Path


stdout_handler = logging.StreamHandler(stream=sys.stdout)
handlers = [stdout_handler]
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
    handlers=handlers
)
logger = logging.getLogger(__name__)


def _load_data(test_data_path: Path):
    ext = test_data_path.suffix
    logger.info(f"file type: {ext}")
    if ext == ".jsonl":
        data = pd.read_json(test_data_path, lines=True, dtype=False)
    elif ext == ".csv":
        data = pd.read_csv(test_data_path)
    else:
        raise Exception("Unsupported file type")
    return data


def _validate_data(data, mlmodel_dict):
    logger.info(f"data cols => {data.columns}")
    # validate model input signature matches with data provided
    if mlmodel_dict.get("signature", None):
        input_signatures_str = mlmodel_dict['signature'].get("inputs", None)
    else:
        logger.warning("signature is missing from MLModel file.")

    if input_signatures_str:
        input_signatures = json.loads(input_signatures_str)
        logger.info(f"input_signatures: {input_signatures}")
        for item in input_signatures:
            if item.get("name") not in data.columns:
                logger.warning(f"Missing {item.get('name')} in test data.")
    else:
        logger.warning("Input signature missing in MLmodel. Prediction might fail.")


def _load_model_and_infer_data(model_dir, data):
    try:
        model = mlflow.pyfunc.load_model(model_dir)
    except Exception as e:
        logger.info(f"Error in loading mlflow model {e}")
        raise Exception(f"Error in loading mlflow model {e}")
    logger.info("Predicting model with test data!!!")
    pred_results = model.predict(data)
    logger.info(f"prediction results\n{pred_results}")


def _get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True, help="Model input path")
    parser.add_argument("--test-data-path", type=Path, required=True, help="Test dataset path")
    return parser


if __name__ == "__main__":
    parser = _get_parser()
    args, _ = parser.parse_known_args()

    model_path: Path = args.model_path
    test_data_path: Path = args.test_dataset

    logger.info("##### logger.info args #####")
    for arg, value in args.__dict__.items():
        logger.info(f"{arg} => {value}")

    MLFLOW_MODEL_FOLDER = "mlflow_model_folder"
    MLMODEL_FILE_NAME = "MLmodel"
    CONDA_YAML_FILE_NAME = "conda.yaml"
    REQUIREMENTS_FILE_NAME = "requirements.txt"
    CONDA_TEST_ENV = "conda_test_env"

    model_dir = f"{model_path}/{MLFLOW_MODEL_FOLDER}"
    mlmodel_file_path = f"{model_dir}/{MLMODEL_FILE_NAME}"
    conda_env_file_path = f"{model_dir}/{CONDA_YAML_FILE_NAME}"
    requirements_file_path = f"{model_dir}/{REQUIREMENTS_FILE_NAME}"

    with open(mlmodel_file_path) as f:
        mlmodel_dict = yaml.safe_load(f)
        logger.info(f"mlmodel :\n{mlmodel_dict}\n")

    with open(conda_env_file_path) as f:
        conda_dict = yaml.safe_load(f)
        logger.info(f"conda :\n{conda_dict}\n")

    conda_env_name = conda_dict.get("name", CONDA_TEST_ENV)
    conda_create_env_command = f"conda env create -n {conda_env_name} -f {conda_env_file_path}"
    conda_env_list = "conda env list"
    conda_activate_env = f"conda activate {conda_env_name}; {conda_env_list}"
    pip_install_cmd = f"pip install -r {requirements_file_path} --user -q"

    logger.info("Installing model pip requirements into current environment")
    exit_code, stdout = run_command(pip_install_cmd)
    if exit_code != 0:
        raise Exception(f"Failure in installing requirements:\n{stdout}\n")
    logger.info(f"Installed pip requirements")

    data = _load_data(test_data_path=test_data_path)
    _validate_data(data, mlmodel_dict)
    _load_model_and_infer_data(model_dir, data)

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Run MLflow Model local validations."""

import argparse
import json
import logging
import mlflow
import shutil
import sys
import pandas as pd
from pathlib import Path
from typing import Dict
from ruamel.yaml import YAML


MLFLOW_MODEL_SCORING_SCRIPT = "validations/mlflow_model_scoring_script.py"
CONDA_ENV_PREFIX = "/opt/conda/envs/inferencing"
MLMODEL_FILE_NAME = "MLmodel"
CONDA_YAML_FILE_NAME = "conda.yaml"

stdout_handler = logging.StreamHandler(stream=sys.stdout)
handlers = [stdout_handler]
logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
    handlers=handlers,
)
logger = logging.getLogger(__name__)


def _load_and_prepare_data(test_data_path: Path, mlmodel: Dict, col_rename_map: Dict) -> pd.DataFrame:
    if not test_data_path:
        return None

    ext = test_data_path.suffix
    logger.info(f"file type: {ext}")
    if ext == ".jsonl":
        data = pd.read_json(test_data_path, lines=True, dtype=False)
    elif ext == ".csv":
        data = pd.read_csv(test_data_path)
    else:
        raise Exception("Unsupported file type")

    # translations
    if col_rename_map:
        data.rename(columns=col_rename_map, inplace=True)

    # Validations
    logger.info(f"data cols => {data.columns}")
    # validate model input signature matches with data provided
    if mlmodel.get("signature", None):
        input_signatures_str = mlmodel['signature'].get("inputs", None)
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
    return data


def _load_and_infer_model(model_dir, data):
    if not data:
        logger.warning("Data not shared. Could not infer the loaded model")
        return

    try:
        model = mlflow.pyfunc.load_model(str(model_dir))
    except Exception as e:
        logger.error(f"Error in loading mlflow model: {e}")
        raise Exception(f"Error in loading mlflow model: {e}")

    try:
        logger.info("Predicting model with test data!!!")
        pred_results = model.predict(data)
        logger.info(f"prediction results\n{pred_results}")
    except Exception as e:
        logger.error(f"Failed to infer model with provided dataset: {e}")
        raise Exception(f"Failed to infer model with provided dataset: {e}")


def _get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True, help="Model input path")
    parser.add_argument("--test-data-path", type=Path, required=False, help="Test dataset path")
    parser.add_argument("--column-rename-map", type=str, required=False, help="")
    parser.add_argument("--output-model-path", type=Path, required=True, help="Output model path")
    return parser


if __name__ == "__main__":
    parser = _get_parser()
    args, _ = parser.parse_known_args()

    model_dir: Path = args.model_path
    test_data_path: Path = args.test_data_path
    col_rename_map_str: str = args.column_rename_map
    output_model_path: Path = args.output_model_path

    logger.info("##### logger.info args #####")
    for arg, value in args.__dict__.items():
        logger.info(f"{arg} => {value}")

    mlmodel_file_path = model_dir / MLMODEL_FILE_NAME
    conda_env_file_path = model_dir / CONDA_YAML_FILE_NAME

    yaml = YAML()

    with open(mlmodel_file_path) as f:
        mlmodel_dict = yaml.load(f)
        logger.info(f"mlmodel :\n{mlmodel_dict}\n")

    with open(conda_env_file_path) as f:
        conda_dict = yaml.load(f)
        logger.info(f"conda :\n{conda_dict}\n")

    col_rename_map = {}
    if col_rename_map_str:
        mapping_list = col_rename_map_str.split(";")
        print(mapping_list)
        for item in mapping_list:
            split = item.split(":")
            if len(split) == 2:
                col_rename_map[split[0].strip()] = split[1].strip()
        logger.info(f"col_rename_map => {col_rename_map}")

    _load_and_infer_model(
        model_dir=model_dir,
        data=_load_and_prepare_data(
            test_data_path=test_data_path,
            mlmodel=mlmodel_dict,
            col_rename_map=col_rename_map,
        ),
    )

    # copy the model to output dir
    shutil.copytree(src=model_dir, dst=output_model_path, dirs_exist_ok=True)

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
import yaml
from applicationinsights import TelemetryClient

tc = TelemetryClient("71b954a8-6b7d-43f5-986c-3d3a6605d803")

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
    tc.track_event(name="FM_import_pipeline_debug_logs",
                   properties={"message": f"file type: {ext}"})
    tc.flush()
    logger.info(f"file type: {ext}")
    if ext == ".jsonl":
        data = pd.read_json(test_data_path, lines=True, dtype=False)
    elif ext == ".csv":
        data = pd.read_csv(test_data_path)
    else:
        tc.track_event(name="FM_import_pipeline_debug_logs",
                       properties={"message": f"Unsupported file type: {ext}"})
        tc.flush()
        raise Exception("Unsupported file type")

    # translations
    if col_rename_map:
        data.rename(columns=col_rename_map, inplace=True)

    # Validations
    tc.track_event(name="FM_import_pipeline_debug_logs",
                   properties={"message": f"data cols => {data.columns}"})
    logger.info(f"data cols => {data.columns}")
    # validate model input signature matches with data provided
    if mlmodel.get("signature", None):
        input_signatures_str = mlmodel['signature'].get("inputs", None)
    else:
        tc.track_event(name="FM_import_pipeline_debug_logs",
                       properties={"message": "signature is missing from MLModel file."})
        logger.warning("signature is missing from MLModel file.")
    tc.flush()

    if input_signatures_str:
        input_signatures = json.loads(input_signatures_str)
        tc.track_event(name="FM_import_pipeline_debug_logs",
                       properties={"message": f"input_signatures: {input_signatures}"})
        logger.info(f"input_signatures: {input_signatures}")
        for item in input_signatures:
            if item.get("name") not in data.columns:
                tc.track_event(name="FM_import_pipeline_debug_logs",
                               properties={"message": f"Missing {item.get('name')} in test data."})
                logger.warning(f"Missing {item.get('name')} in test data.")
    else:
        tc.track_event(name="FM_import_pipeline_debug_logs",
                       properties={"message": "Input signature missing in MLmodel.\
                                   Prediction might fail."})
        logger.warning("Input signature missing in MLmodel. Prediction might fail.")
    tc.flush()
    return data


def _load_and_infer_model(model_dir, data):
    if data is None:
        tc.track_event(name="FM_import_pipeline_debug_logs",
                       properties={"message": "Data not shared. Could not infer the loaded model"})
        tc.flush()
        logger.warning("Data not shared. Could not infer the loaded model")
        return

    try:
        model = mlflow.pyfunc.load_model(str(model_dir))
    except Exception as e:
        tc.track_event(name="FM_import_pipeline_debug_logs",
                       properties={"message": f"Error in loading mlflow model: {e}"})
        tc.flush()
        logger.error(f"Error in loading mlflow model: {e}")
        raise Exception(f"Error in loading mlflow model: {e}")

    try:
        tc.track_event(name="FM_import_pipeline_debug_logs",
                       properties={"message": "Predicting model with test data!!!"})
        logger.info("Predicting model with test data!!!")
        pred_results = model.predict(data)
        tc.track_event(name="FM_import_pipeline_debug_logs",
                       properties={"message": f"prediction results\n{pred_results}"})
        tc.flush()
        logger.info(f"prediction results\n{pred_results}")

    except Exception as e:
        tc.track_event(name="FM_import_pipeline_debug_logs",
                       properties={"message": "Error in predicting model: {e}"})
        tc.flush()
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

    tc.track_event(name="FM_import_pipeline_debug_logs", properties={"message": "##### logger.info args #####"})
    logger.info("##### logger.info args #####")
    for arg, value in args.__dict__.items():
        tc.track_event(name="FM_import_pipeline_debug_logs", properties={"message": f"{arg} => {value}"})
        logger.info(f"{arg} => {value}")
    tc.flush()

    mlmodel_file_path = model_dir / MLMODEL_FILE_NAME
    conda_env_file_path = model_dir / CONDA_YAML_FILE_NAME

    with open(mlmodel_file_path) as f:
        mlmodel_dict = yaml.safe_load(f)
        tc.track_event(name="FM_import_pipeline_debug_logs",
                       properties={"message": f"mlmodel :\n{mlmodel_dict}\n"})
        logger.info(f"mlmodel :\n{mlmodel_dict}\n")

    with open(conda_env_file_path) as f:
        conda_dict = yaml.safe_load(f)
        tc.track_event(name="FM_import_pipeline_debug_logs",
                       properties={"message": f"conda :\n{conda_dict}\n"})
        logger.info(f"conda :\n{conda_dict}\n")

    tc.flush()

    col_rename_map = {}
    if col_rename_map_str:
        mapping_list = col_rename_map_str.split(";")
        print(mapping_list)
        for item in mapping_list:
            split = item.split(":")
            if len(split) == 2:
                col_rename_map[split[0].strip()] = split[1].strip()
        logger.info(f"col_rename_map => {col_rename_map}")
        tc.track_event(name="FM_import_pipeline_debug_logs",
                       properties={"message": f"col_rename_map => {col_rename_map}"})
        tc.flush()

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
    tc.track_event(name="FM_import_pipeline_debug_logs",
                   properties={"message": f"Model copied to {output_model_path}"})
    tc.flush()

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Run Model preprocessor module."""

import argparse
import os
import json
from azureml.model.mgmt.config import ModelFlavor
from azureml.model.mgmt.processors.preprocess import run_preprocess
from azureml.model.mgmt.processors.transformers.config import SupportedTasks
from azureml.model.mgmt.processors.pyfunc.vision.config import Tasks
from azureml.model.mgmt.utils.common_utils import init_tc, tc_log
from pathlib import Path
import shutil


WORKING_DIR = "working_dir"
TMP_DIR = "tmp"


def _get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, required=False, help="Hugging Face model ID")
    parser.add_argument("--task-name", type=str, required=False, help="Hugging Face task type")
    parser.add_argument(
        "--mlflow-flavor",
        type=str,
        default=ModelFlavor.TRANSFORMERS.value,
        help="Model flavor",
    )
    parser.add_argument(
        "--model-download-metadata",
        type=Path,
        required=False,
        help="Model download details",
    )
    parser.add_argument("--model-path", type=Path, required=True, help="Model input path")
    parser.add_argument("--license-file-path", type=Path, required=False, help="License file path")
    parser.add_argument(
        "--mlflow-model-output-dir",
        type=Path,
        required=True,
        help="Output MLFlow model",
    )
    parser.add_argument(
        "--model-import-job-path",
        type=Path,
        required=True,
        help="JSON file containing model job path for model lineage",
    )
    return parser


def _validate_transformers_args(args):
    if not args.get("model_id"):
        tc_log("model_id is a required parameter for hftransformers mlflow flavor.")
        raise Exception("model_id is a required parameter for hftransformers mlflow flavor.")
    if not args.get("task"):
        tc_log("task is a required parameter for hftransformers mlflow flavor.")
        raise Exception("task is a required parameter for hftransformers mlflow flavor.")
    task = args["task"]
    if not SupportedTasks.has_value(task):
        tc_log(f"Unsupported task {task} for hftransformers mlflow flavor.")
        raise Exception(f"Unsupported task {task} for hftransformers mlflow flavor.")


def _validate_pyfunc_args(pyfunc_args):
    if not pyfunc_args.get("task"):
        tc_log("task is a required parameter for pyfunc flavor.")
        raise Exception("task is a required parameter for pyfunc flavor.")
    task = pyfunc_args["task"]
    if not Tasks.has_value(task):
        tc_log(f"Unsupported task {task} for pyfunc flavor.")
        raise Exception(f"Unsupported task {task} for pyfunc flavor.")


if __name__ == "__main__":
    parser = _get_parser()
    args, _ = parser.parse_known_args()
    init_tc()

    model_id = args.model_id
    task_name = args.task_name
    mlflow_flavor = args.mlflow_flavor
    model_download_metadata_path = args.model_download_metadata
    model_path = args.model_path
    mlflow_model_output_dir = args.mlflow_model_output_dir
    model_import_job_path = args.model_import_job_path
    license_file_path = args.license_file_path

    

    if not ModelFlavor.has_value(mlflow_flavor):
        tc_log(f"Unsupported model flavor {mlflow_flavor}")
        raise Exception("Unsupported model flavor")

    preprocess_args = {}
    if model_download_metadata_path:
        with open(model_download_metadata_path) as f:
            download_details = json.load(f)
            preprocess_args.update(download_details.get("tags", {}))
            preprocess_args.update(download_details.get("properties", {}))
    preprocess_args["task"] = task_name if task_name else preprocess_args.get("task")
    preprocess_args["model_id"] = model_id if model_id else preprocess_args.get("model_id")

    tc_log(f"Preprocess args : {preprocess_args}")

    if mlflow_flavor == ModelFlavor.TRANSFORMERS.value:
        _validate_transformers_args(preprocess_args)
    elif mlflow_flavor == ModelFlavor.MMLAB_PYFUNC.value:
        _validate_pyfunc_args(preprocess_args)

    tc_log("Print args")
    tc_log(f"model_id: {model_id}")
    tc_log(f"task_name: {task_name}")
    tc_log(f"mlflow_flavor: {mlflow_flavor}")
    tc_log(f"model_download_metadata_path: {model_download_metadata_path}")
    tc_log(f"model_path: {model_path}")
    tc_log(f"license_file_path: {license_file_path}")

    temp_output_dir = mlflow_model_output_dir / TMP_DIR
    working_dir = mlflow_model_output_dir / WORKING_DIR
    run_preprocess(mlflow_flavor, model_path, working_dir, temp_output_dir, **preprocess_args)
    tc_log("Finished preprocessing")

    # Finishing touches
    shutil.copytree(working_dir, mlflow_model_output_dir, dirs_exist_ok=True)
    shutil.rmtree(working_dir, ignore_errors=True)
    shutil.rmtree(temp_output_dir, ignore_errors=True)

    # Copy license file to output model path
    if license_file_path:
        shutil.copy(license_file_path, mlflow_model_output_dir)

    tc_log(f"listing output directory files: {mlflow_model_output_dir}:\n{os.listdir(mlflow_model_output_dir)}")

    # Add job path
    this_job = os.environ["MLFLOW_RUN_ID"]
    path = f"azureml://jobs/{this_job}/outputs/mlflow_model_folder"
    model_path_dict = {"path": path}
    json_object = json.dumps(model_path_dict, indent=4)
    with open(model_import_job_path, "w") as outfile:
        outfile.write(json_object)
    tc_log("Finished writing job path")

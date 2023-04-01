# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Run Model preprocessor module."""

import argparse
import os
import json
from azureml.model.mgmt.config import ModelFlavor
from azureml.model.mgmt.processors.preprocess import run_preprocess
from pathlib import Path
import shutil


def _get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, required=False, help="Hugging Face model ID")
    parser.add_argument("--task-name", type=str, required=False, help="Hugging Face task type")
    parser.add_argument("--mlflow-flavor", type=str, default=ModelFlavor.TRANSFORMERS.value, help="Model flavor")
    parser.add_argument("--model-download-metadata", type=Path, required=False, help="Model download details")
    parser.add_argument("--model-path", type=Path, required=True, help="Model input path")
    parser.add_argument("--license-file-path", type=Path, required=False, help="License file path")
    parser.add_argument("--mlflow-model-output-dir", type=Path, required=True, help="Output MLFlow model")
    parser.add_argument("--model-job-path", type=Path, required=True,
                        help="JSON file containing model job path for model lineage")
    return parser


def _validate_transformers_args(args):
    if not args.get("model_id"):
        raise Exception("model_id is a required parameter for hftransformers mlflow flavor.")
    if not args.get("task"):
        raise Exception("task is a required parameter for hftransformers mlflow flavor.")


if __name__ == "__main__":
    parser = _get_parser()
    args, _ = parser.parse_known_args()

    model_id = args.model_id
    task_name = args.task_name
    mlflow_flavor = args.mlflow_flavor
    model_download_metadata_path = args.model_download_metadata
    model_path = args.model_path
    mlflow_model_output_dir = args.mlflow_model_output_dir
    model_job_path = args.model_job_path
    license_file_path = args.license_file_path

    print("##### Print args #####")
    for arg, value in args.__dict__.items():
        print(f"{arg} => {value}")

    if not ModelFlavor.has_value(mlflow_flavor):
        raise Exception("Unsupported model flavor")

    preprocess_args = {
        'model_id': model_id,
        'task': task_name,
    }

    with open(model_download_metadata_path) as f:
        download_details = json.load(f)
        preprocess_args.update(download_details.get("tags", {}))
        preprocess_args.update(download_details.get("properties", {}))

    print(preprocess_args)

    if mlflow_flavor == ModelFlavor.TRANSFORMERS.value:
        _validate_transformers_args(preprocess_args)

    run_preprocess(mlflow_flavor, model_path, mlflow_model_output_dir, **preprocess_args)

    # Copy license file in input model_path
    if license_file_path:
        shutil.copy(license_file_path, mlflow_model_output_dir)

    print(f"\nListing mlflow model directory: {mlflow_model_output_dir}:")
    print(os.listdir(mlflow_model_output_dir))

    # Add job path
    this_job = os.environ["MLFLOW_RUN_ID"]
    path = f"azureml://jobs/{this_job}/outputs/mlflow_model_folder"
    model_path_dict = {"path": path}
    json_object = json.dumps(model_path_dict, indent=4)
    with open(model_job_path, "w") as outfile:
        outfile.write(json_object)

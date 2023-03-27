# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Run Model preprocessor module."""

import argparse
import os
import json
from azureml.model.mgmt.config import ModelFlavor
from azureml.model.mgmt.processors.preprocess import run_preprocess
from pathlib import Path


def _get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, required=False, help="Hugging Face model ID")
    parser.add_argument("--task-name", type=str, required=False, help="Hugging Face task type")
    parser.add_argument("--mlflow-flavor", type=str, default=ModelFlavor.TRANSFORMERS.value, help="Model flavor")
    parser.add_argument("--model-download-metadata", type=Path, required=False, help="Model download details")
    parser.add_argument("--model-path", type=Path, required=True, help="Model input path")
    parser.add_argument("--mlflow-model-output-dir", type=Path, required=True, help="Output MLFlow model")
    return parser


def _validate_transformers_args(args):
    if not args.get("model_id"):
        raise Exception("model_id is a required parameter for hftransformers mlflow flavor.")
    if not args.get("task_name"):
        raise Exception("task_name is a required parameter for hftransformers mlflow flavor.")


if __name__ == "__main__":
    parser = _get_parser()
    args, _ = parser.parse_known_args()

    model_id = args.model_id
    task_name = args.task_name
    mlflow_flavor = args.mlflow_flavor
    model_download_metadata_path = args.model_download_metadata
    model_path = args.model_path
    mlflow_model_output_dir = args.mlflow_model_output_dir

    print("##### Print args #####")
    for arg, value in args.__dict__.items():
        print(f"{arg} => {value}")

    if not ModelFlavor.has_value(mlflow_flavor):
        raise Exception("Unsupported model flavor")

    preprocess_args = {
        'model_id': model_id,
        'task_name': task_name,
    }

    with open(model_download_metadata_path) as f:
        download_details = json.load(f)
        preprocess_args.update(download_details.get("tags", {}))
        preprocess_args.update(download_details.get("properties", {}))

    if mlflow_flavor == ModelFlavor.HFTRANSFORMERS.value:
        _validate_transformers_args(preprocess_args)

    run_preprocess(mlflow_flavor, model_path, mlflow_model_output_dir, **preprocess_args)
    print(f"\nListing mlflow model directory: {mlflow_model_output_dir}:")
    print(os.listdir(mlflow_model_output_dir))

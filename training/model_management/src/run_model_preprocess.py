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

LICENSE_FILE = 'LICENSE'

def _get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, required=False, help="Hugging Face model ID")
    parser.add_argument("--task-type", type=str, required=False, help="Hugging Face task type")
    parser.add_argument("--mlflow-flavor", type=str, default=ModelFlavor.HFTRANSFORMERS.value, help="Model flavor")
    parser.add_argument("--model-info", type=Path, required=False, help="Model info path")
    parser.add_argument("--model-path", type=Path, required=True, help="Model input path")
    parser.add_argument("--mlflow-model-output-dir", type=Path, required=True, help="Output MLFlow model")
    return parser


def _validate_hf_transformers_args(args):
    if "model_id" not in args:
        raise Exception("model_id is a required parameter for hftransformers mlflow flavor.")
    if "task_type" not in args:
        raise Exception("task_type is a required parameter for hftransformers mlflow flavor.")


if __name__ == "__main__":
    parser = _get_parser()
    args, _ = parser.parse_known_args()

    model_id = args.model_id
    task_type = args.task_type
    mlflow_flavor = args.mlflow_flavor
    model_info_path = args.model_info
    model_path = args.model_path
    mlflow_model_output_dir = args.mlflow_model_output_dir

    print("##### Print args #####")
    for arg, value in args.__dict__.items():
        print(f"{arg} => {value}")

    if not ModelFlavor.has_value(mlflow_flavor):
        raise Exception("Unsupported model flavor")

    preprocess_args = {
        'model_id': model_id,
        'task_type': task_type,
    }

    with open(str(model_info_path)) as f:
        preprocess_args.update(json.load(f))

    if mlflow_flavor == ModelFlavor.HFTRANSFORMERS.value:
        _validate_hf_transformers_args(preprocess_args)

    run_preprocess(mlflow_flavor, model_path, mlflow_model_output_dir, **preprocess_args)

    ## Adding support for license file if exist in model path
    if os.path.exists(Path(model_path,LICENSE_FILE)):
        shutil.copy(Path(model_path,LICENSE_FILE), mlflow_model_output_dir)
    else:
        print(f"License file does not exist for :{args.model_id}")

    print(f"\nListing mlflow model directory: {mlflow_model_output_dir}:")
    print(os.listdir(mlflow_model_output_dir))

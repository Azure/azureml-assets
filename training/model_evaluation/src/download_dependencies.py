# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Script for downloading and installing mlflow model dependencies."""

from mlflow.pyfunc import _get_model_dependencies
from argparse import ArgumentParser
import traceback
try:
    from pip import main as pipmain
except ImportError:
    from pip._internal import main as pipmain
REQUIREMENTS_FILE = "./requirements.txt"


def main():
    """Download and Install mlflow model dependencies."""
    parser = ArgumentParser()
    parser.add_argument("--model-uri", type=str, dest="model_uri", required=False, default="")
    parser.add_argument("--mlflow-model", type=str, dest="mlflow_model", required=False, default=None)

    args = parser.parse_args()

    model_uri = args.model_uri.strip()
    mlflow_model = args.mlflow_model
    if mlflow_model:
        model_uri = mlflow_model

    reqs_file = _get_model_dependencies(model_uri, "pip")

    with open(reqs_file, "r") as f:
        for line in f.readlines():
            if line.strip() == "mlflow":
                continue
            if "azureml_evaluate_mlflow" in line.strip() or "azureml-evaluate-mlflow" in line.strip():
                continue
            if "azureml_metrics" in line.strip() or "azureml-metrics" in line.strip():
                continue
            try:
                pipmain(["install", line.strip()])
            except Exception:
                print("Failed to install package", line)
                print("Traceback:")
                traceback.print_exc()
    pipmain(["install", "--upgrade", "mltable"])


if __name__ == "__main__":
    main()

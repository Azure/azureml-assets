# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Script for downloading and installing mlflow model dependencies."""

from mlflow.pyfunc import _get_model_dependencies
from argparse import ArgumentParser
import traceback
from accelerate import PartialState

distributed_state = PartialState()

try:
    from pip import main as pipmain
except ImportError:
    from pip._internal import main as pipmain
REQUIREMENTS_FILE = "./requirements.txt"

UPDATE_PACKAGES = [
    # "transformers",
    # "accelerate"
]


def to_be_updated(package):
    """Get packages to be updated.

    Args:
        package (str)

    Returns:
        _type_: bool
    """
    # TODO: See if there are any packages that should be updated
    # Right now skipping all extra packages
    return False
    # return any([update_package_name in package.strip() for update_package_name in UPDATE_PACKAGES])


def main():
    """Download and Install mlflow model dependencies."""
    parser = ArgumentParser()
    parser.add_argument("--mlflow-model", type=str, dest="mlflow_model", required=False, default=None)

    args = parser.parse_args()

    mlflow_model = args.mlflow_model
    reqs_file = _get_model_dependencies(mlflow_model, "pip")

    with open(reqs_file, "r") as f:
        for line in f.readlines():
            if not to_be_updated(line.strip()):
                print("Skipping Package", line.strip())
                continue
            try:
                pipmain(["install", line.strip()])
            except Exception:
                print("Failed to install package", line)
                print("Traceback:")
                traceback.print_exc()

    # Install Telemetry packages
    pipmain(["install", "--upgrade", "azureml-core"])
    pipmain(["install", "--upgrade", "azureml-telemetry"])
    pipmain(["install", "--upgrade", "mltable"])
    pipmain(["freeze"])


if __name__ == "__main__":
    if distributed_state.is_local_main_process:
        main()
    distributed_state.wait_for_everyone()

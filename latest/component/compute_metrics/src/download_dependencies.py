# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Script for downloading and installing mlflow model dependencies."""

from mlflow.pyfunc import _get_model_dependencies
from utils import ArgumentParser
from logging_utilities import (
    custom_dimensions, get_logger, log_traceback, swallow_all_exceptions, get_azureml_exception
)
from azureml.telemetry.activity import log_activity
from error_definitions import InvalidModel
from exceptions import ModelValidationException
from validation import _validate_model
from constants import ArgumentLiterals, TelemetryConstants

import sys
import subprocess

custom_dimensions.app_name = TelemetryConstants.DOWNLOAD_MODEL_DEPENDENCIES
logger = get_logger(name=__name__)
custom_dims_dict = vars(custom_dimensions)


@swallow_all_exceptions(logger)
def main():
    """Download and Install mlflow model dependencies."""
    parser = ArgumentParser()
    parser.add_argument("--mlflow-model", type=str, dest=ArgumentLiterals.MLFLOW_MODEL, required=True)

    args, _ = parser.parse_known_args()
    args = vars(args)

    _validate_model(args)

    with log_activity(logger, TelemetryConstants.DOWNLOAD_MODEL_DEPENDENCIES,
                      custom_dimensions=custom_dims_dict):
        try:
            reqs_file = _get_model_dependencies(args[ArgumentLiterals.MLFLOW_MODEL], "pip")
        except Exception as e:
            exception = get_azureml_exception(ModelValidationException, InvalidModel, e)
            log_traceback(exception, logger)
            raise exception

        ignore_deps = [
            "mlflow",
            "azureml_evaluate_mlflow",
            "azureml-evaluate-mlflow",
            "azureml_metrics",
            "azureml-metrics",
            ]
        install_deps = [
            "azure-ai-ml",
        ]
        logger.info(f"Reading from {reqs_file}")
        with open(reqs_file, "r") as f:
            for line in f.readlines():
                if any(dep in line.strip() for dep in ignore_deps):
                    continue
                install_deps += [line.strip()]

        no_install = []
        if len(install_deps) > 0:
            try:
                command = [sys.executable, '-m', 'pip', 'install'] + install_deps
                command_str = ' '.join(command)
                logger.info(f"Installing dependencies. Executing command: {command_str}")
                subprocess.check_output(command, stderr=subprocess.STDOUT)
            except Exception as e:
                logger.warning(f"Installing dependencies failed with error:\n{e.stdout.decode('utf-8')}")
                logger.info("Installing dependencies one by one.")

                command = [sys.executable, '-m', 'pip', 'install', 'dep']
                for dep in install_deps:
                    try:
                        command[-1] = dep
                        command_str = ' '.join(command)
                        logger.info(f"Installing dependencies. Executing command: {command_str}")
                        subprocess.check_output(command, stderr=subprocess.STDOUT)
                    except Exception as e:
                        logger.warning(f"Installing dependency {dep} failed with error:\n{e.stdout.decode('utf-8')}")
                        no_install += [dep]

        if len(no_install):
            no_install = "\n".join(no_install)
            logger.warning(
                f"Could not install following dependencies - \n{no_install}"
                f"\nSome functionality might not work as expected."
            )


if __name__ == "__main__":
    main()

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Script for downloading and installing mlflow model dependencies."""

from mlflow.pyfunc import _get_model_dependencies
from utils import ArgumentParser
from logging_utilities import (
    custom_dimensions, get_logger, log_traceback, swallow_all_exceptions, get_azureml_exception
)
from azureml.telemetry.activity import log_activity
from error_definitions import DownloadDependenciesError
from exceptions import ModelValidationException
from validation import _validate_model
from constants import ArgumentLiterals, TelemetryConstants

try:
    from pip import main as pipmain
except ImportError:
    from pip._internal import main as pipmain
REQUIREMENTS_FILE = "./requirements.txt"

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

    reqs_file = _get_model_dependencies(args[ArgumentLiterals.MLFLOW_MODEL], "pip")

    ignore_deps = [
        "mlflow",
        "azureml_evaluate_mlflow",
        "azureml-evaluate-mlflow",
        "azureml_metrics",
        "azureml-metrics"
    ]
    install_deps = []
    with log_activity(logger, TelemetryConstants.DOWNLOAD_MODEL_DEPENDENCIES,
                      custom_dimensions=custom_dims_dict):
        with open(reqs_file, "r") as f:
            for line in f.readlines():
                if any(dep in line.strip() for dep in ignore_deps):
                    continue
                install_deps += [line.strip()]

        if len(install_deps) > 0:
            try:
                pipmain(["install"] + install_deps)
            except Exception as e:
                message_kwargs = {"dependencies": ' '.join(install_deps)}
                exception = get_azureml_exception(ModelValidationException, DownloadDependenciesError, e,
                                                  error=repr(e), **message_kwargs)
                log_traceback(exception, logger)
                raise exception


if __name__ == "__main__":
    main()

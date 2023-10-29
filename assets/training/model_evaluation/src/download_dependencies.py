# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Script for downloading and installing mlflow model dependencies."""

from mlflow.pyfunc import _get_model_dependencies
from utils import ArgumentParser
from exceptions import swallow_all_exceptions
from logging_utilities import custom_dimensions, get_logger, log_traceback
from azureml.telemetry.activity import log_activity
import constants
# import traceback
from error_definitions import DownloadDependenciesError
from exceptions import ModelValidationException
from azureml._common._error_definition.azureml_error import AzureMLError

try:
    from pip import main as pipmain
except ImportError:
    from pip._internal import main as pipmain
REQUIREMENTS_FILE = "./requirements.txt"

logger = get_logger(name=__name__)
custom_dimensions.app_name = constants.TelemetryConstants.COMPONENT_NAME
custom_dims_dict = vars(custom_dimensions)


@swallow_all_exceptions(logger)
def main():
    """Download and Install mlflow model dependencies."""
    parser = ArgumentParser()
    parser.add_argument("--mlflow-model", type=str, dest="mlflow_model", required=True)

    args, _ = parser.parse_known_args()

    reqs_file = _get_model_dependencies(args.mlflow_model, "pip")

    ignore_deps = [
        "mlflow",
        "azureml_evaluate_mlflow",
        "azureml-evaluate-mlflow",
        "azureml_metrics",
        "azureml-metrics"
    ]
    install_deps = []
    with log_activity(logger, constants.TelemetryConstants.DOWNLOAD_MODEL_DEPENDENCIES,
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
                exception = ModelValidationException._with_error(
                    AzureMLError.create(DownloadDependenciesError, error=repr(e),
                                        message_kwargs={"dependencies": ' '.join(install_deps)}),
                    inner_exception=e
                )
                log_traceback(exception, logger)


if __name__ == "__main__":
    main()

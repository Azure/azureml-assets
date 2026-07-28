# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""File containing function for FTaaS data import component."""

from azureml.acft.common_components import (
    get_logger_app,
)
from azureml.acft.common_components.utils.error_handling.swallow_all_exceptions_decorator import (
    swallow_all_exceptions,
)
from azureml.telemetry.activity import log_activity

from azure.ai.ml import Input
from mldesigner import Output, command_component

from common.constants import (
    TelemetryConstants,
)

logger = get_logger_app(
    "azureml.acft.contrib.hf.nlp.entry_point.data_import.data_import"
)


@command_component
@swallow_all_exceptions(logger)
def validate(
    validation_file_path: Input(type="string", optional=True),  # noqa: F821
) -> Output(type="boolean", is_control=True):  # noqa: F821
    """Entry function of model validation script."""
    with log_activity(
        logger,
        TelemetryConstants.VERSION_SELECTION,
        {"validation_file_path": validation_file_path},
    ):
        logger.info("Validating arguments: " + repr(validation_file_path))
        if validation_file_path:
            return True
        else:
            return False

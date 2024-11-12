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
    DataGenerationTaskType,
    TelemetryConstants,
)

logger = get_logger_app(
    "azureml.acft.contrib.hf.nlp.entry_point.data_import.data_import"
)


@command_component
@swallow_all_exceptions(logger)
def validate(
    data_generation_task_type: Input(type="string", optional=False),  # noqa: F821
) -> Output(type="boolean", is_control=True):  # noqa: F821
    """Entry function of model validation script."""
    with log_activity(
        logger,
        TelemetryConstants.VERSION_SELECTION,
        {"data_generation_task_type": data_generation_task_type},
    ):
        logger.info("Validating arguments: " + repr(data_generation_task_type))
        if data_generation_task_type == DataGenerationTaskType.CONVERSATION:
            return False
        else:
            return True

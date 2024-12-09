# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Select Model Framework Component."""

from azure.ai.ml import Input
from mldesigner import Output, command_component
from azureml.model.mgmt.utils.logging_utils import get_logger
from azureml.model.mgmt.utils.exceptions import swallow_all_exceptions

logger = get_logger(__name__)

@command_component
@swallow_all_exceptions(logger)
def validate(model_framework: Input(type="string", optional=False))  -> bool:
    """Entry function of model validation script."""

    if model_framework == "MMLab":
        result = True
    else:
        result = False

    logger.info(f"Model framework: {model_framework}, result: {result}")

    return result

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Validate Import Pipeline Parameters."""
# import argparse

from azureml.model.mgmt.utils.exceptions import swallow_all_exceptions
from azureml.model.mgmt.utils.logging_utils import get_logger
from mldesigner import Input, Output, command_component

logger = get_logger(name=__name__)

@command_component
@swallow_all_exceptions(logger)
def validate(
        model_framework: Input(type="string", optional=False, default='HuggingFace')
        ) -> Output(type="boolean", is_control=True): # noqa: F821
    """Return true if model_framework is MMLab, False otherwise."""

    print(f"Model framework: {model_framework}")
    result = model_framework == "MMLab"
    
    print(f"Model framework: {model_framework}, result: {result}")
    logger.info(f"Model framework: {model_framework}, result: {result}")
    
    # Return the result as a boolean control output
    return result

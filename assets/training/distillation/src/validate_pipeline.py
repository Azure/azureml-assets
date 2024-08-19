# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Script for validating distillation pipeline arguments."""
import logging
from argparse import ArgumentParser

from azureml.acft.contrib.hf import VERSION, PROJECT_NAME
from azureml.acft.contrib.hf.nlp.constants.constants import LOGS_TO_BE_FILTERED_IN_APPINSIGHTS
from azureml.acft.common_components import get_logger_app, set_logging_parameters, LoggingLiterals
from azureml.acft.common_components.utils.error_handling.swallow_all_exceptions_decorator import (
    swallow_all_exceptions,
)

from generate_data import get_parser

logger = get_logger_app(
    "azureml.acft.distillation.scripts.components.validate_pipeline"
)

COMPONENT_NAME = "oss_distillation_validate_pipeline"

def update_parser(parser: ArgumentParser):
    """
    Updates parser with flags from finetuning task as part of the distillation 
    pipeline.
    """
    # TODO (nandakumars): add relevant arguments.
    return parser

@swallow_all_exceptions(time_delay=5)
def run():
    set_logging_parameters(
        task_type="DistillationPipelineValidation",
        acft_custom_dimensions={
            LoggingLiterals.PROJECT_NAME: PROJECT_NAME,
            LoggingLiterals.PROJECT_VERSION_NUMBER: VERSION,
            LoggingLiterals.COMPONENT_NAME: COMPONENT_NAME
        },
        azureml_pkg_denylist_logging_patterns=LOGS_TO_BE_FILTERED_IN_APPINSIGHTS,
        log_level=logging.info
    )
    
    # Get data generation component input parameters.
    parser = get_parser()
    parser = update_parser(parser=parser)
    args, _ = parser.parse_known_args()


if __name__ == "main":
    run()
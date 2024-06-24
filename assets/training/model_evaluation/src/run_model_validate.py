# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Validate Model Evaluation Pipeline Parameters."""

from constants import ArgumentLiterals, TelemetryConstants
from logging_utilities import get_logger, custom_dimensions, swallow_all_exceptions
from validation import (
    validate_model_prediction_args,
    validate_common_args,
    validate_and_get_columns,
    validate_compute_metrics_label_column_arg,
)
import os

from azureml.automl.core.shared.logging_utilities import mark_path_as_loggable
from azureml.telemetry.activity import log_activity
from mldesigner import Input, Output, command_component

# Mark current path as allowed
mark_path_as_loggable(os.path.dirname(__file__))

custom_dimensions.app_name = TelemetryConstants.TRIGGER_VALIDATION_NAME
logger = get_logger(name=__name__)
custom_dims_dict = vars(custom_dimensions)


@command_component
@swallow_all_exceptions(logger)
def validate(
        task: Input(type="string", optional=False),  # noqa: F821
        data: Input(type="uri_folder", optional=False),  # noqa: F821
        mlflow_model: Input(type="mlflow_model", optional=False, default=None),  # noqa: F821
        label_column_name: Input(type="string", optional=True, default=None),  # noqa: F821
        input_column_names: Input(type="string", optional=True, default=None),  # noqa: F821
        device: Input(type="string", optional=False),  # noqa: F821
        batch_size: Input(type="integer", optional=True, default=None),  # noqa: F821
        config_file_name: Input(type="uri_file", optional=True, default=None),  # noqa: F821
        config_str: Input(type="string", optional=True, default=None),  # noqa: F821
) -> Output(type="boolean", is_control=True):  # noqa: F821
    """Entry function of model validation script."""
    if label_column_name:
        label_column_name = label_column_name.split(",")

    if input_column_names:
        input_column_names = [i.strip() for i in input_column_names.split(",") if i and not i.isspace()]

    args = {
        ArgumentLiterals.TASK: task,
        ArgumentLiterals.DATA: data,
        ArgumentLiterals.MLFLOW_MODEL: mlflow_model,
        ArgumentLiterals.LABEL_COLUMN_NAME: label_column_name,
        ArgumentLiterals.INPUT_COLUMN_NAMES: input_column_names,
        ArgumentLiterals.DEVICE: device,
        ArgumentLiterals.BATCH_SIZE: batch_size,
        ArgumentLiterals.CONFIG_FILE_NAME: config_file_name,
        ArgumentLiterals.CONFIG_STR: config_str,
    }

    with log_activity(logger, TelemetryConstants.VALIDATION_NAME,
                      custom_dimensions=custom_dims_dict):
        logger.info("Validating arguments: " + repr(args))
        validate_common_args(args)
        validate_model_prediction_args(args)
        validate_compute_metrics_label_column_arg(args)

        _ = validate_and_get_columns(args)

    return True

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Contains helper functions for logging."""

from typing import Any, Dict
import logging
import os
import hashlib
import mlflow

from azureml.core import Run


def log_mlflow_params(**kwargs: Any) -> None:
    """
    Log the provided key-value pairs as parameters in MLflow.

    If a file path or a list of file paths is provided in the value, the checksum of the
    file(s) is calculated and logged as the parameter value. If `None` is provided as
    the value, the parameter is not logged.

    :param **kwargs: Key-value pairs of parameters to be logged in MLflow.
    :return: None
    """
    MLFLOW_PARAM_VALUE_MAX_LEN = 500
    OVERFLOW_STR = '...'
    params = {}
    for key, value in kwargs.items():
        if isinstance(value, str) and os.path.isfile(value):
            # calculate checksum of input dataset
            checksum = hashlib.sha256(open(value, "rb").read()).hexdigest()
            params[key] = checksum
        elif isinstance(value, list) and all(isinstance(item, str) and os.path.isfile(item) for item in value):
            # calculate checksum of input dataset
            checksum = hashlib.sha256(b"".join(open(item, "rb").read() for item in value)).hexdigest()
            params[key] = checksum
        else:
            if value is not None:
                if isinstance(value, str) and len(value) > MLFLOW_PARAM_VALUE_MAX_LEN:
                    value_len = MLFLOW_PARAM_VALUE_MAX_LEN - len(OVERFLOW_STR)
                    params[key] = value[: value_len] + OVERFLOW_STR
                else:
                    params[key] = value

    mlflow.log_params(params)


def get_logger(filename: str) -> logging.Logger:
    """
    Create and configure a logger based on the provided filename.

    This function creates a logger with the specified filename and configures it
    by setting the logging level to INFO, adding a StreamHandler to the logger,
    and specifying a specific log message format.

    :param filename: The name of the file associated with the logger.
    :return: The configured logger.
    """
    logger = logging.getLogger(filename)
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    logger.addHandler(stream_handler)

    formatter = logging.Formatter(
        "[%(asctime)s - %(name)s - %(levelname)s] - %(message)s"
    )
    stream_handler.setFormatter(formatter)
    return logger

logger = get_logger(__name__)

def log_params_and_metrics(
    parameters: Dict[str, Any],
    metrics: Dict[str, Any],
    log_to_parent: bool,
) -> None:
    """Log mlflow params and metrics to current run and parent run."""
    filtered_metrics = {}
    for key in metrics:
        if isinstance(metrics[key], bool):
            # For bool value, latest version of mlflow throws an error.
            filtered_metrics[key] = float(metrics[key])
        elif isinstance(metrics[key], (int, float)):
            filtered_metrics[key] = metrics[key]
    # Log to current run
    logger.info(
        f"Attempting to log {len(parameters)} parameters and {len(filtered_metrics)} metrics."
    )
    try:
        log_mlflow_params(**parameters)
    except Exception as ex:
        logger.error(f"Failed to log parameters to current run due to {ex}")
    try:
        mlflow.log_metrics(filtered_metrics)
    except Exception as ex:
        logger.error(f"Failed to log metrics to current run due to {ex}")
    if log_to_parent:
        # Log to parent run
        try:
            parent_run_id = Run.get_context().parent.id
            ml_client = mlflow.tracking.MlflowClient()
            for param_name, param_value in parameters.items():
                param_value_to_log = param_value
                if isinstance(param_value, str) and len(param_value) > 500:
                    param_value_to_log = param_value[: 497] + '...'
                try:
                    ml_client.log_param(parent_run_id, param_name, param_value_to_log)
                except Exception as ex:
                    logger.error(f"Failed to log parameter {param_name} to root run due to {ex}.")
            for metric_name, metric_value in filtered_metrics.items():
                try:
                    ml_client.log_metric(parent_run_id, metric_name, metric_value)
                except Exception as ex:
                    logger.error(f"Failed to log metric {metric_name} to root run due to {ex}.")
        except Exception as ex:
            logger.error(f"Failed to log parameters and metrics to root run due to {ex}.")
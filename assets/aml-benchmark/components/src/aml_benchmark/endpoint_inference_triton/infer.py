# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Infer File."""

from typing import List, Dict, Any, Tuple
import datetime

import numpy as np
import gevent.ssl
import tritonclient.http as httpclient
from azureml._common._error_definition.azureml_error import AzureMLError

from aml_benchmark.utils.exceptions import swallow_all_exceptions, BenchmarkValidationException
from aml_benchmark.utils.error_definitions import BenchmarkValidationError
from aml_benchmark.utils.aml_run_utils import get_authorization_header
from aml_benchmark.utils.constants import TaskType


HTTPS = "https://"
AUTHORIZATION = "authorization"
BEARER = "Bearer"
PREDICTION = "prediction"
START_TIME_ISO = "start_time_iso"
END_TIME_ISO = "end_time_iso"
TIME_TAKEN_MS = "time_taken_ms"
BATCH_SIZE = "batch_size"


def _get_headers(connections_name: str, deployment_name: str) -> Dict[str, str]:
    """Get headers."""
    return {
        **get_authorization_header(connections_name),
        "Content-Type": "application/json",
        "azureml-model-deployment": deployment_name,
    }

def get_predictions(
    task_type: str,
    endpoint_url: str,
    deployment_name: str,
    model_name: str,
    model_version: str,
    connections_name: str,
    payload: List[List[httpclient.InferInput]],
) -> Tuple[List[Dict[str, str]], List[Dict[str, Any]]]:
    """Get predictions and perf data."""
    headers = _get_headers(connections_name, deployment_name)
    client = httpclient.InferenceServerClient(
        url=endpoint_url.replace(HTTPS, ""),
        ssl=True,
        ssl_context_factory=gevent.ssl._create_default_https_context
    )

    # Check status of triton server
    health_ctx = client.is_server_ready(headers=headers)
    if not health_ctx:
        raise BenchmarkValidationException._with_error(
            AzureMLError.create(BenchmarkValidationError, error_details="Triton server is not ready.")
        )

    # Check status of model
    status_ctx = client.is_model_ready(model_name, model_version, headers)
    if not status_ctx:
        raise BenchmarkValidationException._with_error(
            AzureMLError.create(BenchmarkValidationError, error_details="Model is not ready.")
        )

    results = []
    perf_data: List[Dict[str, Any]] = []
    batch_size = 1
    for input in payload:
        start_time_iso = datetime.datetime.now().isoformat()
        results.append(client.infer(model_name, inputs=input, headers=headers))
        end_time_iso = datetime.datetime.now().isoformat()

        time_taken_ms = round((
            datetime.datetime.fromisoformat(end_time_iso) - datetime.datetime.fromisoformat(start_time_iso)
        ).total_seconds() * 1000, 3)
        perf_data.append({
            START_TIME_ISO: start_time_iso,
            END_TIME_ISO: end_time_iso,
            TIME_TAKEN_MS: time_taken_ms,
            BATCH_SIZE: batch_size
        })

    if task_type == TaskType.TEXT_GENERATION.value:
        for i, result in enumerate(results):
            result_str = result.as_numpy("outputs")[0][0].decode("utf-8")
            results[i] = {PREDICTION: result_str}
    elif task_type == TaskType.CHAT_COMPLETION.value:
        for i, result in enumerate(results):
            result_str = "".join(
                [val.decode("utf-8") for val in result.as_numpy("text_output").tolist()]
            )
            results[i] = {PREDICTION: result_str}

    return results, perf_data
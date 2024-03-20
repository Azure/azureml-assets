# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Constants for Benchmarking."""
from enum import Enum


AML_BENCHMARK_DYNAMIC_LOGGER_ENTRY_POINT = "aml_benchmark.azureml_benchmark_custom_logger"

class AuthenticationType(Enum):
    """Authentication Type enum for endpoints."""

    AZUREML_WORKSPACE_CONNECTION = "azureml_workspace_connection"
    MANAGED_IDENTITY = "managed_identity"


class ModelType(Enum):
    """Model Type enum."""

    OAI = "oai"
    OSS = "oss"
    VISION_OSS = "vision_oss"


AOAI_ENDPOINT_DOMAIN_SUFFIX_LIST = [
    "openai.azure.com",
    "api.cognitive.microsoft.com",
    "cognitiveservices.azure.com"
]
MIR_ENDPOINT_DOMAIN_SUFFIX_LIST = ["inference.ml.azure.com"]
SERVERLESS_ENDPOINT_DOMAIN_SUFFIX_LIST = ["inference.ai.azure.com"]

_URL_TYPES_MAPPING = {
    "azure_openai": AOAI_ENDPOINT_DOMAIN_SUFFIX_LIST,
    "azureml_online_endpoint": MIR_ENDPOINT_DOMAIN_SUFFIX_LIST,
    "azureml_serverless_endpoint": SERVERLESS_ENDPOINT_DOMAIN_SUFFIX_LIST
}
_DEFAULT_URL_TYPE = "azureml_online_endpoint"


def get_endpoint_type(url: str) -> str:
    """
    Get the endpoint type for a given endpoint URL.

    :param url: The URL of the endpoint.
    :return: The type of the endpoint.
    """
    for url_type, url_suffix_list in _URL_TYPES_MAPPING.items():
        for url_suffix in url_suffix_list:
            if url_suffix in url:
                return url_type
    return _DEFAULT_URL_TYPE

class TelemetryConstants:
    """Telemetry Constants."""

    APP_INSIGHT_HANDLER_NAME = "AppInsightsHandler"

    NON_PII_MESSAGE = '[Hidden as it may contain PII]'
    DEFAULT_APP_NAME = "AML_BENCHMARK"
    DEFAULT_VERSION = "Unknown"


class ExceptionTypes:
    """AzureML Exception Types."""

    User = "User"
    System = "System"
    Service = "Service"
    Unclassified = "Unclassified"
    All = {User, System, Service, Unclassified}


KNOWN_MODULES = {
    'batch_benchmark_score',
    "batch_config_generator",
    "batch_inference_preparer",
    "batch_output_formatter",
    "batch_resource_manager",
    "benchmark_result_aggregator",
    "compute_performance_metrics",
    "dataset_downloader",
    "dataset_preprocessor",
    "dataset_sampler",
    "inference_postprocessor",
    "prompt_crafter"
}
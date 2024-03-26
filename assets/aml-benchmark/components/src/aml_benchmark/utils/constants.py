# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Constants for Benchmarking."""
from enum import Enum


class AuthenticationType(Enum):
    """Authentication Type enum for endpoints."""

    AZUREML_WORKSPACE_CONNECTION = "azureml_workspace_connection"
    MANAGED_IDENTITY = "managed_identity"


class ModelType(Enum):
    """Model Type enum."""

    OAI = "oai"
    OSS = "oss"
    VISION_OSS = "vision_oss"


class LoggerConfig:
    """Logger Config."""

    AML_BENCHMARK_HANDLER_NAME = "AMLBenchmarkHandler"
    APPINSIGHT_HANDLER_NAME = "AppInsightsHandler"
    DEFAULT_MODULE_NAME = "aml_benchmark"
    VERBOSITY_LEVEL = "DEBUG"
    OFFLINE_RUN_MESSAGE = "OFFLINE_RUN"
    ASSET_NOT_FOUND = "AssetID missing in run details"
    NON_PII_MESSAGE = '[Hidden as it may contain PII]'


class ExceptionTypes:
    """AzureML Exception Types."""

    User = "User"
    System = "System"
    Service = "Service"
    Unclassified = "Unclassified"
    All = {User, System, Service, Unclassified}


ROOT_RUN_PROPERTIES = {
    "PipelineType": "Benchmark"
}


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

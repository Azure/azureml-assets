# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Constants for Benchmarking."""
from enum import Enum


class Constants:
    """Constants for benchmarking."""

    MAX_RETRIES = 7
    MAX_RETRIES_OAI = 1
    BASE_DELAY = 10
    MAX_DELAY = 600
    MAX_THREADS = 20
    BACKOFF_FACTOR = 2
    MAX_TIMEOUT_SEC = 180
    RETRIABLE_STATUS_CODES = {413, 429, 500, 502, 503, 504, None}


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


class IntermediateNames:
    """Names for intermediate files and directories."""

    DATASTORE_DIRECTORY_URL_TEMPLATE = "AmlDatastore://{datastore_name}/{directory_name}"
    RANDOM_IMAGE_DIRECTORY_TEMPLATE = "images/{random_id}"
    IMAGE_FILE_NAME_TEMPLATE = "image_{image_counter:09d}.png"


ROOT_RUN_PROPERTIES = {
    "PipelineType": "Benchmark"
}


AOAI_ENDPOINT_DOMAIN_SUFFIX_LIST = [
    "openai.azure.com",
    "api.cognitive.microsoft.com",
    "cognitiveservices.azure.com",
]
MIR_ENDPOINT_DOMAIN_SUFFIX_LIST = ["inference.ml.azure.com"]
SERVERLESS_ENDPOINT_DOMAIN_SUFFIX_LIST = ["inference.ai.azure.com", "models.ai.azure.com"]

_URL_TYPES_MAPPING = {
    "azure_openai": AOAI_ENDPOINT_DOMAIN_SUFFIX_LIST,
    "azureml_online_endpoint": MIR_ENDPOINT_DOMAIN_SUFFIX_LIST,
    "azureml_serverless_endpoint": SERVERLESS_ENDPOINT_DOMAIN_SUFFIX_LIST,
}
_DEFAULT_URL_TYPE = "azureml_online_endpoint"


class ApiType():
    """Api Type."""

    Unknown = 'unknown'
    Completion = 'completion'
    ChatCompletion = 'chat_completion'


COMPLETION_API_SUFFIX_LIST = ["v1/completions"]
CHAT_COMPLETION_API_SUFFIX_LIST = ["v1/chat/completions"]
DEFAULT_API_TYPE = ApiType.Completion
API_TYPE_MAPPING = {
    ApiType.Completion: COMPLETION_API_SUFFIX_LIST,
    ApiType.ChatCompletion: CHAT_COMPLETION_API_SUFFIX_LIST
}


def get_api_type(url: str) -> str:
    """Get the api type for a given endpoint URL.

    :param url: The URL of the endpoint.
    :return: API type of the endpoint.
    """
    return next((
        api_type for api_type, suffixes in API_TYPE_MAPPING.items()
        if any(suffix in url for suffix in suffixes)),
        DEFAULT_API_TYPE
    )


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


DATASET_CONFIG_2_NAME_MAP = {
    ("formal_logic,high_school_european_history,high_school_us_history,"
     "high_school_world_history,international_law,jurisprudence,logical_fallacies,"
     "moral_disputes,moral_scenarios,philosophy,prehistory,professional_law,world_religions"): "mmlu_humanities",
    ("business_ethics,clinical_knowledge,college_medicine,global_facts,human_aging,management,"
     "marketing,medical_genetics,miscellaneous,nutrition,professional_accounting,"
     "professional_medicine,virology"): "mmlu_other",
    ("econometrics,high_school_geography,high_school_government_and_politics,high_school_macroeconomics,"
     "high_school_microeconomics,high_school_psychology,human_sexuality,professional_psychology,public_relations,",
     "security_studies,sociology,us_foreign_policy"): "mmlu_social_sciences",
    ("abstract_algebra,anatomy,astronomy,college_biology,college_chemistry,college_computer_science,"
     "college_mathematics,college_physics,computer_security,conceptual_physics,electrical_engineering,"
     "elementary_mathematics,high_school_biology,high_school_chemistry,high_school_computer_science,"
     "high_school_mathematics,high_school_physics,high_school_statistics,machine_learning"): "mmlu_stem",
}

REQUIRED_TELEMETRY_KEYS_MAP = {
        "downloader.dataset_name": "dataset_name",
        "endpoint.endpoint_url": "endpoint_url",
        "endpoint.deployment_name": "deployment_name",
        "endpoint.model_type": "model_type",
        "endpoint.authentication_type": "authentication_type"
    }

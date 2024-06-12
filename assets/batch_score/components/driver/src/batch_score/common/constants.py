# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Constants."""

COMPONENT_NAME_KEY = "component_name"
COMPONENT_VERSION_KEY = "component_version"
METADATA_JSON_FILENAME = "metadata.json"
OS_ENVIRON_HOST_NAME = "HostName"
OS_ENVIRON_WORKSPACE = "AZUREML_ARM_WORKSPACE_NAME"
OS_ENVIRON_RUN_ID = "AZUREML_RUN_ID"
OS_ENVIRON_NODE_RANK = "AZUREML_CR_NODE_RANK"
OS_ENVIRON_COMPUTE_CONTEXT = "AZUREML_CR_COMPUTE_CONTEXT"
TRAFFIC_GROUP = "batch"

# HTTP request timeout related environment variables
BATCH_SCORE_BACKOFF_FACTOR_REQUEST_TIMEOUT_ENV_VAR = "BATCH_SCORE_BACKOFF_FACTOR_REQUEST_TIMEOUT"
BATCH_SCORE_INITIAL_REQUEST_TIMEOUT_ENV_VAR = "BATCH_SCORE_INITIAL_REQUEST_TIMEOUT"
BATCH_SCORE_MAX_REQUEST_TIMEOUT_ENV_VAR = "BATCH_SCORE_MAX_REQUEST_TIMEOUT"

# Other environment variables
BATCH_SCORE_EMIT_PROMPTS_TO_JOB_LOG_ENV_VAR = "BATCH_SCORE_EMIT_PROMPTS_TO_JOB_LOG"
BATCH_SCORE_NO_DEPLOYMENTS_BACK_OFF_ENV_VAR = "BATCH_SCORE_NO_DEPLOYMENTS_BACK_OFF"
BATCH_SCORE_POLL_DURING_NO_DEPLOYMENTS_ENV_VAR = "BATCH_SCORE_POLL_DURING_NO_DEPLOYMENTS"
BATCH_SCORE_SURFACE_TELEMETRY_EXCEPTIONS_ENV_VAR = "BATCH_SCORE_SURFACE_TELEMETRY_EXCEPTIONS"
BATCH_SCORE_TRACE_LOGGING_ENV_VAR = "BATCH_SCORE_TRACE_LOGGING"

COMPLETION_API_TYPE = "completion"
CHAT_COMPLETION_API_TYPE = "chat_completion"
VESTA_API_TYPE = "vesta"
VESTA_CHAT_COMPLETION_API_TYPE = "vesta_chat_completion"
EMBEDDINGS_API_TYPE = "embeddings"

DV_COMPLETION_API_PATH = "v1/engines/davinci/completions"
DV_EMBEDDINGS_API_PATH = "v1/engines/davinci/embeddings"
DV_CHAT_COMPLETIONS_API_PATH = "v1/engines/davinci/chat/completions"
VESTA_RAINBOW_API_PATH = "v1/rainbow"
VESTA_CHAT_COMPLETIONS_API_PATH = "v1/rainbow/chat/completions"

DEFAULT_REQUEST_PATHS = (DV_COMPLETION_API_PATH,
                         DV_EMBEDDINGS_API_PATH,
                         DV_CHAT_COMPLETIONS_API_PATH,
                         VESTA_RAINBOW_API_PATH,
                         VESTA_CHAT_COMPLETIONS_API_PATH)

AOAI_ENDPOINT_DOMAIN_SUFFIX_LIST = ["openai.azure.com", "api.cognitive.microsoft.com", "cognitiveservices.azure.com"]
MIR_ENDPOINT_DOMAIN_SUFFIX = "inference.ml.azure.com"
SERVERLESS_ENDPOINT_DOMAIN_SUFFIX = "inference.ai.azure.com"

CONNECTION_AUTH_TYPE = "connection"
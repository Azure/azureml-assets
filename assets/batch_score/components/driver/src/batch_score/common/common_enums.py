# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Configuration Enums."""

from strenum import StrEnum


class EndpointType(StrEnum):
    """Endpoint Type."""

    AOAI = 'AOAI'
    BatchPool = 'BatchPool'
    Serverless = 'Serverless'
    MIR = 'MIR'


class AuthenticationType(StrEnum):
    """Authentication Type."""

    Unknown = 'unknown'
    ManagedIdentity = 'managed_identity'
    ApiKey = 'api_key'
    WorkspaceConnection = 'azureml_workspace_connection'


class ApiType(StrEnum):
    """Api Type."""

    Unknown = 'unknown'
    Completion = 'completion'
    ChatCompletion = 'chat_completion'
    Vesta = 'vesta'
    VestaChatCompletion = 'vesta_chat_completion'
    Embedding = 'embedding'

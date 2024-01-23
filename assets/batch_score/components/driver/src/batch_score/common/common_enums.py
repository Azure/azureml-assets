# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from strenum import StrEnum


class EndpointType(StrEnum):
    AOAI = 'AOAI'
    BatchPool = 'BatchPool'
    Serverless = 'Serverless'
    MIR = 'MIR'


class AuthenticationType(StrEnum):
    Unknown = 'unknown'
    ManagedIdentity = 'managed_identity'
    ApiKey = 'api_key'
    WorkspaceConnection = 'azureml_workspace_connection'


class ApiType(StrEnum):
    Unknown = 'unknown'
    Completion = 'completion'
    ChatCompletion = 'chat_completion'
    Vesta = 'vesta'
    VestaChatCompletion = 'vesta_chat_completion'
    Embedding = 'embedding'

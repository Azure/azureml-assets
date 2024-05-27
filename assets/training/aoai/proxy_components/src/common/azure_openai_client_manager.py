# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Azure OpenAI client manager."""

import requests
from azure.identity import ManagedIdentityCredential
from azure.ai.ml.identity import AzureMLOnBehalfOfCredential
from azure.mgmt.cognitiveservices import CognitiveServicesManagementClient
from azure.mgmt.cognitiveservices.models import ApiKeys
from azure.core.pipeline.policies import BearerTokenCredentialPolicy
from azure.core.pipeline import PipelineRequest, PipelineContext
from azure.core.rest import HttpRequest
from enum import Enum
from openai import AzureOpenAI
import os
from typing import Optional
from common.logging import get_logger

logger = get_logger(__name__)


class AuthenticationType(Enum):
    """Enum for authentication type."""
    MANAGED_IDENTITY = "managed_identity"
    USER_IDENTITY = "user_identity"


class AzureOpenAIClientManager:
    """Class for **authentication** related information used for the run."""

    ENV_CLIENT_ID_KEY = "DEFAULT_IDENTITY_CLIENT_ID"
    MLFLOW_TRACKING_URI = "MLFLOW_TRACKING_URI"
    api_version = "2024-04-01-preview"

    def __init__(self, endpoint_name, endpoint_resource_group: Optional[str], endpoint_subscription: Optional[str], authentication_type: Optional[str]):
        """Initialize the AzureOpenAIClientManager."""
        self.endpoint_name = endpoint_name
        self.endpoint_resource_group = endpoint_resource_group
        self.endpoint_subscription = endpoint_subscription
        workspace_subscription, workspace_resource_group = self._get_workspace_subscription_id_resource_group()
        if endpoint_subscription is None:
            logger.info("AOAI resource subscription id is empty, will default to workspace subscription")
            self.endpoint_subscription = workspace_subscription
            if self.endpoint_subscription is None:
                raise Exception("endpoint_subscription is None")

        if endpoint_resource_group is None:
            logger.info("AOAI resource resource group is empty, will default to workspace resource group")
            self.endpoint_resource_group = workspace_resource_group
            if self.endpoint_resource_group is None:
                raise Exception("endpoint_resource_group is None")
        
        if authentication_type is None:
            logger.info("Authentication type is not provided, will default to 'managed_identity'")
            self.authentication_type = AuthenticationType.MANAGED_IDENTITY
        else:
            self.authentication_type = AuthenticationType(authentication_type)
        self.aoai_client = self._get_azure_openai_client()

    def _get_client_id(self) -> str:
        """Get the client id."""
        return os.environ.get(AzureOpenAIClientManager.ENV_CLIENT_ID_KEY, None)

    def _get_workspace_subscription_id_resource_group(self) -> str:
        """Get current subscription id."""
        uri = os.environ.get(AzureOpenAIClientManager.MLFLOW_TRACKING_URI, None)
        if uri is None:
            return None, None
        uri_segments = uri.split("/")
        subscription_id = uri_segments[uri_segments.index("subscriptions") + 1]
        resource_group = uri_segments[uri_segments.index("resourceGroups") + 1]
        return subscription_id, resource_group

    def _get_credential(self) -> ManagedIdentityCredential:
        """Get the credential."""
        if self.authentication_type == AuthenticationType.MANAGED_IDENTITY:
            return ManagedIdentityCredential(
                client_id=self._get_client_id())
        elif self.authentication_type == AuthenticationType.USER_IDENTITY:
            return AzureMLOnBehalfOfCredential()
        else:
            raise ValueError(f"Wrong authentication type: {self.authentication_type}")

    def get_key_from_cognitive_service_account(self, client: CognitiveServicesManagementClient) -> str:
        """Get key from cognitive service account."""
        api_keys: ApiKeys = client.accounts.list_keys(resource_group_name=self.endpoint_resource_group,
                                                      account_name=self.endpoint_name)
        return api_keys.key1

    def get_endpoint_from_cognitive_service_account(self, client: CognitiveServicesManagementClient) -> str:
        """Get endpoint from cognitive service account."""
        account = client.accounts.get(resource_group_name=self.endpoint_resource_group,
                                      account_name=self.endpoint_name)
        logger.info("Endpoint: {}".format(account.properties.endpoint))
        return account.properties.endpoint

    def _get_bearer_token_provider(self, credential):
        policy = BearerTokenCredentialPolicy(credential, "https://cognitiveservices.azure.com/.default")

        def _make_request():
            return PipelineRequest(HttpRequest("CredentialWrapper", "https://fakeurl"), PipelineContext(None))

        def wrapper() -> str:
            request = _make_request()
            policy.on_request(request)
            return request.http_request.headers["Authorization"][len("Bearer ") :]
        return wrapper

    def _get_azure_openai_client(self) -> AzureOpenAI:
        """Get azure openai client."""
        if self.authentication_type == AuthenticationType.MANAGED_IDENTITY:
            logger.info("Trying to get azure openai client using managed identity")
            if self._get_client_id() is None:
                logger.info("Managed identity client id is empty, will fail...")
                raise Exception("Managed identity client id is empty")
            else:
                logger.info("Managed identity client id is set, will use managed identity authentication")
                client = CognitiveServicesManagementClient(credential=self._get_credential(),
                                                           subscription_id=self.endpoint_subscription)
                return AzureOpenAI(azure_endpoint=self.get_endpoint_from_cognitive_service_account(client),
                                   api_key=self.get_key_from_cognitive_service_account(client),
                                   api_version=AzureOpenAIClientManager.api_version)
        elif self.authentication_type == AuthenticationType.USER_IDENTITY:
            logger.info("Trying to get azure openai client using user identity")
            credential = self._get_credential()
            client = CognitiveServicesManagementClient(credential=credential,
                                                       subscription_id=self.endpoint_subscription)
            return AzureOpenAI(azure_endpoint=self.get_endpoint_from_cognitive_service_account(client),
                               azure_ad_token_provider=self._get_bearer_token_provider(credential),
                               api_version=AzureOpenAIClientManager.api_version)
        else:
            raise ValueError(f"Wrong authentication type: {self.authentication_type}")

    @property
    def data_upload_url(self) -> str:
        """Url to call for uploading data to AOAI resource."""
        base_url = self.aoai_client.base_url  # https://<aoai-resource-name>.openai.azure.com/openai/
        return f"{base_url}/files/import?api-version={self.api_version}"

    def _get_auth_header(self) -> dict:
        return {"api-key": self.aoai_client.api_key,
                "Content-Type": "application/json"}

    def upload_data_to_aoai(self, body: dict[str, str]):
        """Upload data to aoai via rest call."""
        try:
            logger.info(f"Uploading data to endpoint: {self.data_upload_url} via rest call")
            resp = requests.post(self.data_upload_url, headers=self._get_auth_header(), json=body)
            logger.info(f"Recieved response status : {resp.status_code}, value: {resp.text}")
            return resp.text
        except requests.exceptions.RequestException as e:
            logger.error(f"Got Exception : {e} while uploading data to AOAI resource")

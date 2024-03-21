# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Azure OpenAI client manager."""

from azure.identity import ManagedIdentityCredential
from azure.mgmt.cognitiveservices import CognitiveServicesManagementClient
from azure.mgmt.cognitiveservices.models import ApiKeys
from openai import AzureOpenAI

import os
from typing import Optional
from common.logging import get_logger

logger = get_logger(__name__)


class AzureOpenAIClientManager:
    """Class for **authentication** related information used for the run."""

    ENV_CLIENT_ID_KEY = "DEFAULT_IDENTITY_CLIENT_ID"
    MLFLOW_TRACKING_URI = "MLFLOW_TRACKING_URI"
    api_version = "2023-12-01-preview"

    def __init__(self, endpoint_name, endpoint_resource_group: Optional[str], endpoint_subscription: Optional[str]):
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

    def _get_client_id(self) -> str:
        """Get the client id."""
        return os.environ.get(AzureOpenAIClientManager.ENV_CLIENT_ID_KEY, None)

    def _get_workspace_subscription_id_resource_group(self) -> str:
        """Get current subscription id."""
        uri = os.environ.get(AzureOpenAIClientManager.MLFLOW_TRACKING_URI, None)
        if uri is None:
            return None
        uri_segments = uri.split("/")
        subscription_id = uri_segments[uri_segments.index("subscriptions") + 1]
        resource_group = uri_segments[uri_segments.index("resourceGroups") + 1]
        return subscription_id, resource_group

    def _get_credential(self) -> ManagedIdentityCredential:
        """Get the credential."""
        credential = ManagedIdentityCredential(
            client_id=self._get_client_id())
        return credential

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

    def get_azure_openai_client(self) -> AzureOpenAI:
        """Get azure openai client."""
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

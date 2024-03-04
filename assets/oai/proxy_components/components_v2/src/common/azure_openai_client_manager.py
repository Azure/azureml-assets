# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

""" Azure OpenAI client manager."""

from azureml.core import Run
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
    api_version = "2023-12-01-preview"

    def __init__(self, endpoint_name, endpoint_resource_group: Optional[str], endpoint_subscription: Optional[str]):
        """ Initialize the AzureOpenAIClientManager."""
        self.endpoint_name = endpoint_name
        self.endpoint_resource_group = endpoint_resource_group
        self.endpoint_subscription = endpoint_subscription
        self.run_context = Run.get_context()
        if endpoint_subscription is None:
            logger.info("AOAI resource subscription id is empty, will default to workspace subscription")
            self.endpoint_subscription = self.run_context.experiment.workspace.subscription_id
        if endpoint_resource_group is None:
            logger.info("AOAI resource resource group is empty, will default to workspace resource group")
            self.endpoint_resource_group = self.run_context.experiment.workspace.resource_group

    def _get_client_id(self) -> str:
        """Get the client id."""
        return os.environ.get(AzureOpenAIClientManager.ENV_CLIENT_ID_KEY, None)

    def _get_credential(self) -> ManagedIdentityCredential:
        """Get the credential."""
        credential = ManagedIdentityCredential(
            client_id=self._get_client_id())
        return credential

    def get_key_from_cognitive_service_account(self, client: CognitiveServicesManagementClient) -> str:
        """ Gets key from cognitive service account"""
        api_keys: ApiKeys = client.accounts.list_keys(resource_group_name=self.endpoint_resource_group,
                                                      account_name=self.endpoint_name)
        return api_keys.key1

    def get_endpoint_from_cognitive_service_account(self, client: CognitiveServicesManagementClient) -> str:
        """ Gets endpoint from cognitive service account"""
        account = client.accounts.get(resource_group_name=self.endpoint_resource_group,
                                      account_name=self.endpoint_name)
        logger.info("Endpoint: {}".format(account.properties.endpoint))
        return account.properties.endpoint

    def get_azure_openai_client(self) -> AzureOpenAI:
        """ Gets azure openai client"""
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

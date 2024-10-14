# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Azure OpenAI client manager."""

import requests
from azure.identity import ManagedIdentityCredential
from azure.mgmt.cognitiveservices import CognitiveServicesManagementClient
from azure.mgmt.cognitiveservices.models import ApiKeys
from openai import AzureOpenAI
import os
from typing import Optional, Dict
from common.logging import get_logger
from exception_handler import retry_on_exception

logger = get_logger(__name__)


class AzureOpenAIClientManager:
    """Class for **authentication** related information used for the run."""

    ENV_CLIENT_ID_KEY = "DEFAULT_IDENTITY_CLIENT_ID"
    MLFLOW_TRACKING_URI = "MLFLOW_TRACKING_URI"
    api_version = "2024-04-01-preview"

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

    def _get_azure_openai_client(self) -> AzureOpenAI:
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

    @property
    def data_upload_url(self) -> str:
        """Url to call for uploading data to AOAI resource."""
        base_url = self.aoai_client.base_url  # https://<aoai-resource-name>.openai.azure.com/openai/
        return f"{base_url}/files/import?api-version={self.api_version}"

    def _get_auth_header(self) -> dict:
        return {"api-key": self.aoai_client.api_key,
                "Content-Type": "application/json"}

    @retry_on_exception
    def upload_data_to_aoai(self, body: dict[str, str]):
        """Upload data to aoai via rest call."""
        try:
            logger.info(f"Uploading data to endpoint: {self.data_upload_url} via rest call")
            resp = requests.post(self.data_upload_url, headers=self._get_auth_header(), json=body)
            logger.info(f"Recieved response status : {resp.status_code}, value: {resp.text}")
            return resp.text
        except requests.exceptions.RequestException as e:
            logger.error(f"Got Exception : {e} while uploading data to AOAI resource")


    @retry_on_exception
    def submit_finetune_job(self, 
                            model, 
                            hyperparameters: Dict[str, str],
                            hyperparameters_1p: Dict[str, str],
                            training_file_id: str,
                            validation_file_id: str,
                            suffix=None):
        """Submit fine-tune job to AOAI."""
        logger.debug(f"Starting fine-tune job, model: {model}, suffix: {suffix},\
                     training_file_id: {training_file_id}, validation_file_id: {validation_file_id}")

        finetune_job = self.aoai_client.fine_tuning.jobs.create(
            model=model,
            training_file=training_file_id,
            validation_file=validation_file_id,
            hyperparameters=hyperparameters,
            extra_headers=hyperparameters_1p,
            suffix=suffix)
        logger.debug(f"started finetuning job in Azure OpenAI resource. Job id: {finetune_job.id}")
        logger.debug(f"Response of finetune create call : {str(finetune_job)}")
        return finetune_job.id
    
    @retry_on_exception
    def retrieve_job(self, job_id: str):
        """Retrieve fine-tune job."""
        return self.aoai_client.fine_tuning.jobs.retrieve(job_id)
    
    @retry_on_exception
    def get_file_content(self, file_id: str):
        return self.aoai_client.files.content(file_id=file_id)

    @retry_on_exception
    def list_events(self, job_id: str):
        return self.aoai_client.fine_tuning.jobs.list_events(job_id).data
    
    @retry_on_exception
    def upload_file(self, file_name, file_data):
        return self.aoai_client.files.create(file=(file_name, file_data, 'application/json'), purpose='fine-tune')
    
    @retry_on_exception
    def wait_for_processing(self, file_id: str):
        return self.aoai_client.files.wait_for_processing(file_id)

    @retry_on_exception
    def cancel_job(self, job_id: str):
        return self.aoai_client.fine_tuning.jobs.cancel(job_id)
    @retry_on_exception
    def delete_file(self, file_id: str):
        return self.aoai_client.files.delete(file_id)

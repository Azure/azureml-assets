# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Copied from https://github.com/Azure/azureai-insiders/tree/main/previews/batch-inference-using-aoai
# and then modified.

import json
import os
from abc import abstractmethod
from datetime import datetime, timezone
from ..common_enums import EndpointType

import requests
from azure.core.credentials import AccessToken
from azure.identity import (
    CredentialUnavailableError,
    DefaultAzureCredential,
    ManagedIdentityCredential,
)
from azureml.core import Run
from requests.adapters import HTTPAdapter
from urllib3 import Retry

from ..telemetry.logging_utils import get_logger


class AuthProvider:

    @abstractmethod
    def get_auth_headers(self) -> dict:
        pass


class IdentityAuthProvider(AuthProvider):

    SCOPE_COGNITIVE = "https://cognitiveservices.azure.com/.default"

    def __init__(
            self,
            use_user_identity: bool,
            managed_identity_client_id: str = None):

        # The identity used to score an AOAI deployment must have
        # the role "Cognitive Services User" on the AOAI resource.
        # The role "Contributor" is not sufficient.

        if managed_identity_client_id and managed_identity_client_id != "DEFAULT_IDENTITY_CLIENT_ID":
            get_logger().info(f"Using managed identity credential with client id {managed_identity_client_id}")
            self.__credential = ManagedIdentityCredential(client_id=managed_identity_client_id)
        else:
            client_id = os.environ.get('DEFAULT_IDENTITY_CLIENT_ID', None)

            if client_id:
                get_logger().info(f"Using managed identity credential with default client id {client_id}")
                self.__credential = ManagedIdentityCredential(client_id=client_id)
            else:
                get_logger().info("Using default azure credential.")
                self.__credential = DefaultAzureCredential()

        self.__access_token: AccessToken = None

    def get_auth_headers(self) -> dict:
        return {
            'Authorization': 'Bearer ' + self.__get_access_token(),
        }

    def __get_access_token(self) -> AccessToken:
        # If there's a token that isn't expired, return that
        if not self.__is_token_expired():
            return self.__access_token.token

        try:
            get_logger().info("Attempting to get token from MSI")
            self.__access_token = self.__credential.get_token(IdentityAuthProvider.SCOPE_COGNITIVE)
        except CredentialUnavailableError:
            get_logger().info("Failed to get token from MSI")

        return self.__access_token.token

    def __is_token_expired(self) -> bool:
        return not self.__access_token or \
            self.__access_token.expires_on <= datetime.now(timezone.utc).timestamp() + (5 * 60)


class MissingApiKeyNameException(Exception):
    def __init__(self):
        pass

    def __str__(self):
        return ("'api_key_name' cannot be empty when using the authentication_type 'api_key'. "
                "Please set 'api_key_name' to the name of the key vault secret that contains the API key "
                "of the scoring_url. The secret must be placed in the key vault that is linked to the "
                "AzureML workspace in which the batch scoring job runs.")


class ApiKeyAuthProvider(AuthProvider):

    def __init__(
            self,
            api_key_name) -> None:

        if api_key_name is None:
            ex = MissingApiKeyNameException()
            get_logger().error(str(ex))
            raise ex

        self.__api_key_name = api_key_name
        self.__api_key = self.__get_api_key()

    def get_auth_headers(self) -> dict:
        return {
            'api-key': self.__api_key,
        }

    def __get_api_key(self) -> str:
        run = Run.get_context()

        return run.get_secret(self.__api_key_name)


class WorkspaceConnectionAuthProvider(AuthProvider):

    def __init__(self, connection_name, endpoint_type) -> None:
        self._current_workspace = None
        self._connection_name = connection_name
        self._endpoint_type = endpoint_type

    @property
    def current_workspace(self):
        """Get the current workspace."""
        if self._current_workspace is None:
            self._current_workspace = Run.get_context().experiment.workspace
        return self._current_workspace

    def get_auth_headers(self) -> dict:
        """Get the auth headers."""
        resp = self._get_workspace_connection_by_name()
        return {
            EndpointType.AOAI: {'api-key': resp['properties']['credentials']['key']},
            EndpointType.MIR: {'Authorization': f"Bearer {resp['properties']['credentials']['key']}"},
            EndpointType.Serverless: {'Authorization': resp['properties']['credentials']['key']},
        }[self._endpoint_type]

    def _get_workspace_connection_by_name(self) -> dict:
        """Get a workspace connection from the workspace."""
        if hasattr(self.current_workspace._auth, "get_token"):
            bearer_token = self.current_workspace._auth.get_token(
                "https://management.azure.com/.default").token
        else:
            bearer_token = self.current_workspace._auth.token

        endpoint = self.current_workspace.service_context._get_endpoint("api")

        url_list = [
            endpoint,
            "rp/workspaces/subscriptions",
            self.current_workspace.subscription_id,
            "resourcegroups",
            self.current_workspace.resource_group,
            "providers",
            "Microsoft.MachineLearningServices",
            "workspaces",
            self.current_workspace.name,
            "connections",
            self._connection_name,
            "listsecrets?api-version=2023-02-01-preview"
        ]
        response = self._send_post_request('/'.join(url_list), {
            "Authorization": f"Bearer {bearer_token}",
            "content-type": "application/json"
        }, {})

        return response.json()

    def _send_post_request(self, url: str, headers: dict, payload: dict):
        """Send a POST request."""
        with self._create_session_with_retry() as session:
            response = session.post(url, data=json.dumps(payload), headers=headers)
            response.raise_for_status()

        return response

    def _create_session_with_retry(self, retry: int = 3) -> requests.Session:
        """Create requests.session with retry."""
        retry_policy = self._get_retry_policy(num_retry=retry)

        session = requests.Session()
        session.mount("https://", HTTPAdapter(max_retries=retry_policy))
        session.mount("http://", HTTPAdapter(max_retries=retry_policy))
        return session

    def _get_retry_policy(self, num_retry: int = 3) -> Retry:
        """Request retry policy with increasing backoff."""
        status_forcelist = [413, 429, 500, 502, 503, 504]
        backoff_factor = 0.4
        retry_policy = Retry(
            total=num_retry,
            read=num_retry,
            connect=num_retry,
            backoff_factor=backoff_factor,
            status_forcelist=status_forcelist,
            raise_on_status=False
        )
        return retry_policy

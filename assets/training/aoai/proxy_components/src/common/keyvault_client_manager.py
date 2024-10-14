# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Key Vault Client manager."""

from azure.identity import ManagedIdentityCredential
import os
from azure.ai.ml import MLClient
from azure.keyvault.secrets import SecretClient
from exception_handler import retry_on_exception
from common.logging import get_logger

logger = get_logger(__name__)


class KeyVaultClientManager:
    """Class for **authentication** related information used for the run."""

    ENV_CLIENT_ID_KEY = "DEFAULT_IDENTITY_CLIENT_ID"
    MLFLOW_TRACKING_URI = "MLFLOW_TRACKING_URI"

    def __init__(self):
        """Initialize the KeyVaultClientManager."""
        subscription_id, resource_group, workspace_name = self._get_workspace_resource_group_subscription_id()
        self.subscription_id = subscription_id
        self.resource_group = resource_group
        self.workspace_name = workspace_name

    def _get_ml_client(self) -> MLClient:
        credential = self._get_credential()
        ml_client = MLClient(credential, self.subscription_id, self.resource_group, self.workspace_name)
        return ml_client

    def _get_client_id(self) -> str:
        """Get the client id."""
        return os.environ.get(KeyVaultClientManager.ENV_CLIENT_ID_KEY, None)

    def _get_workspace_resource_group_subscription_id(self):
        """Get current subscription id."""
        uri = os.environ.get(KeyVaultClientManager.MLFLOW_TRACKING_URI, None)
        if uri is None:
            raise ValueError("mlflow tracking uri not set")
        uri_segments = uri.split("/")
        subscription_id = uri_segments[uri_segments.index("subscriptions") + 1]
        resource_group = uri_segments[uri_segments.index("resourceGroups") + 1]
        workspace = uri_segments[uri_segments.index("workspaces") + 1]

        return subscription_id, resource_group, workspace

    def _get_credential(self) -> ManagedIdentityCredential:
        """Get the credential."""
        credential = ManagedIdentityCredential(
            client_id=self._get_client_id())
        return credential

    def _get_workspace_keyvault_name(self):
        ml_client = self._get_ml_client()
        workspace = ml_client.workspaces.get(self.workspace_name)
        key_vault_path = workspace.key_vault
        key_vault_name = os.path.basename(key_vault_path)
        return key_vault_name

    @property
    def keyvault_name(self) -> str:
        """Name for the Keyvault associated with user workspace."""
        return self._get_workspace_keyvault_name()

    @property
    def keyvault_url(self) -> str:
        """Url for the Keyvault associated with user workspace."""
        return f"https://{self.keyvault_name}.vault.azure.net/"

    def get_keyvault_client(self):
        """Client for the Keyvault associated with user workspace."""
        return SecretClient(credential=self._get_credential(), vault_url=self.keyvault_url)


    @retry_on_exception
    def get_secret_from_keyvault(self, key: str) -> str:
        keyvault_client = self.get_keyvault_client()
        logger.info(f"fetching key: {key} from keyvault: {self.keyvault_name}")
        return keyvault_client.get_secret(key).value
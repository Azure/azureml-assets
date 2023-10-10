# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Class for AOAI online endpoint."""


from typing import Optional
from azureml._model_management._util import get_requests_session
from azureml._common._error_definition.azureml_error import AzureMLError
from utils.error_definitions import BenchmarkValidationError
from utils.exceptions import BenchmarkValidationException
from .online_endpoint import OnlineEndpoint, ResourceState
from .online_endpoint_model import OnlineEndpointModel
from utils.logging import get_logger


logger = get_logger(__name__)


class AOAIOnlineEndpoint(OnlineEndpoint):
    """Class for AOAI online endpoint."""

    def __init__(
            self,
            workspace_name: str,
            resource_group: str,
            subscription_id: str,
            online_endpoint_model: OnlineEndpointModel,
            online_endpoint_url: Optional[str] = None,
            endpoint_name: Optional[str] = None,
            deployment_name: Optional[str] = None,
            sku: Optional[str] = None,
            location: Optional[str] = None,
            api_version: str = '2023-05-01',
            connections_name: Optional[str] = None
    ):
        """Init method."""
        # For AOAI endpoint, the account name is the same as the endpoint name.
        super().__init__(
            workspace_name,
            resource_group,
            subscription_id,
            online_endpoint_url,
            endpoint_name,
            deployment_name,
            sku,
            online_endpoint_model,
            connections_name
        )
        self._api_version = api_version
        self._location = location
        self._validate_settings()

    @property
    def scoring_url(self) -> str:
        """Get the scoring url."""
        if self._online_endpoint_url is None:
            return self._get_scoring_url()
        return self._online_endpoint_url

    def get_endpoint_name_from_url(self) -> str:
        """Get the endpoint."""
        return self._online_endpoint_url.split('/')[2].split('.')[0]

    def get_deployment_name_from_url(self) -> str:
        """Get the deployment name."""
        return self._online_endpoint_url.split('/')[5]

    def endpoint_state(self) -> ResourceState:
        """Check if the endpoint exists."""
        resp = self._call_endpoint(get_requests_session().get, self._aoai_account_url)
        logger.info("Calling {} returned {} with content {}.".format(
            self._aoai_deployment_url, resp.status_code, self._get_content_from_response(resp)))
        return self._get_resource_state(resp)

    def deployment_state(self) -> ResourceState:
        """Check if the deployment exists."""
        resp = self._call_endpoint(get_requests_session().get, self._aoai_deployment_url)
        logger.info("Calling {} returned {} with content {}.".format(
            self._aoai_deployment_url, resp.status_code, self._get_content_from_response(resp)))
        return self._get_resource_state(resp)

    def create_endpoint(self):
        """Create the endpoint."""
        self._validate_model(validate_version=False)
        payload = {
            "location": self._location,
            "kind": "OpenAI",
            "sku": {
                "name": 'S0'
            },
            "properties": {},
            "identity": {
                "type": "SystemAssigned"
            }
        }
        resp = self._call_endpoint(get_requests_session().put, self._aoai_account_url, payload=payload)
        self._raise_if_not_success(resp)
        logger.info("Calling {} returned {} with content {}.".format(
            self._aoai_deployment_url, resp.status_code, self._get_content_from_response(resp)))

    def create_deployment(self):
        """Create deployment."""
        self._validate_model(validate_version=False)
        payload = {
            "sku":
            {
                "name": "Standard",
                "capacity": self.sku
            },
            "properties":
            {
                "model":
                {
                    "format": "OpenAI",
                    "name": self._model.model_name,
                    "version": self._model.model_version
                },
                "raiPolicyName": "Microsoft.Default",
                "versionUpgradeOption": "OnceNewDefaultVersionAvailable"
            }
        }
        resp = self._call_endpoint(get_requests_session().put, self._aoai_deployment_url, payload=payload)
        self._raise_if_not_success(resp)
        logger.info("Calling {} returned {} with content {}.".format(
            self._aoai_deployment_url, resp.status_code, self._get_content_from_response(resp)))

    def get_endpoint_authorization_header(self) -> dict:
        """Get the authorization header."""
        return {'api-key': self._get_endpoint_token()}

    def _get_endpoint_token(self) -> str:
        resp = self._call_endpoint(get_requests_session().post, self._aoai_deployment_list_key_url)
        self._raise_if_not_success(
            resp,
            msg=f'Failed to get the keys using {self._aoai_deployment_list_key_url}.'
                f'response code: {resp.status_code}, response content: {resp.content}.'
        )
        keys_content = self._get_content_from_response(resp)
        return keys_content['key1']

    def get_resource_authorization_header(self) -> dict:
        """Get the authorization header."""
        return {'Authorization': f'Bearer {self._get_resource_token()}'} if self._credential else {}

    def delete_endpoint(self):
        """Delete the endpoint."""
        resp = self._call_endpoint(get_requests_session().delete, self._aoai_account_url)
        self._raise_if_not_success(resp)
        logger.info("Calling {} returned {} with content {}.".format(
            self._aoai_deployment_url, resp.status_code, self._get_content_from_response(resp)))

    def delete_deployment(self):
        """Delete the deployment."""
        resp = self._call_endpoint(get_requests_session().delete, self._aoai_deployment_url)
        self._raise_if_not_success(resp)
        logger.info("Calling {} returned {} with content {}.".format(
            self._aoai_deployment_url, resp.status_code, self._get_content_from_response(resp)))

    @property
    def sku(self) -> str:
        """Get the sku."""
        return int(self._sku) if self._sku else 120

    @property
    def _aoai_base_url(self) -> str:
        """Get the base url for aoai."""
        url_list = [
            self._arm_base_url, 'providers/Microsoft.CognitiveServices'

        ]
        return "/".join(url_list)

    @property
    def _aoai_account_base_url(self) -> str:
        """Get the base url for aoai account."""
        url_list = [
            self._aoai_base_url, 'accounts', self.endpoint_name
        ]
        return "/".join(url_list)

    @property
    def _aoai_deployment_base_url(self) -> str:
        """Get the base url for aoai deployment."""
        url_list = [
            self._aoai_account_base_url, 'deployments', self.deployment_name
        ]
        return "/".join(url_list)

    @property
    def _aoai_deployment_list_key_url(self) -> str:
        """Aoai deployment list key url."""
        return f'{self._aoai_account_base_url}/listKeys?api-version={self._api_version}'

    @property
    def _aoai_account_url(self) -> str:
        """Aoai account url."""
        return f'{self._aoai_account_base_url}?api-version={self._api_version}'

    @property
    def _aoai_deployment_url(self) -> str:
        """Aoai deployment url."""
        return f'{self._aoai_deployment_base_url}?api-version={self._api_version}'

    def _get_scoring_url(self) -> str:
        """Get the scoring url."""
        url_list = [
            f'https://{self.endpoint_name}.openai.azure.com', 'openai', 'deployments',
            self.deployment_name, 'chat', 'completions?api-version=2023-07-01-preview'
        ]
        return '/'.join(url_list)

    def _validate_settings(self) -> None:
        """Validate settings."""
        super()._validate_settings()
        try:
            if self._sku is not None:
                _ = int(self._sku)
        except ValueError:
            raise BenchmarkValidationException._with_error(
                AzureMLError.create(
                    BenchmarkValidationError,
                    error_details='SKU for AOAI model must be int.')
                )

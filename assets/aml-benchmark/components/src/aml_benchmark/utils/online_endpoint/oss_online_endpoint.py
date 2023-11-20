# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Class for OSS online endpoint."""


from typing import Optional, Union
import time
from azureml._model_management._util import get_requests_session
from azureml._model_management._util import _get_mms_url

from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment
)
from azure.ai.ml.entities import (
    OnlineRequestSettings,
    ProbeSettings,
)

from azureml._common._error_definition.azureml_error import AzureMLError
from .online_endpoint import OnlineEndpoint, ResourceState
from .online_endpoint_model import OnlineEndpointModel

from ..error_definitions import BenchmarkValidationError
from ..exceptions import BenchmarkValidationException
from ..logging import get_logger


LOGGER = get_logger(__name__)


class OSSOnlineEndpoint(OnlineEndpoint):
    """Class for OSS online endpoint."""

    REQUEST_TIMEOUT_MS = 90000

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
            api_version: str = '2023-04-01-Preview',
            connections_name: str = None,
            additional_deployment_env_vars = {},
            deployment_env: str = None
    ):
        """Init method."""
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
        self._additional_deployment_env_vars = additional_deployment_env_vars
        self._deployment_env = deployment_env

    def get_endpoint_name_from_url(self) -> str:
        """Get the endpoint."""
        return self._online_endpoint_url.split('/')[2].split('.')[0]

    def get_deployment_name_from_url(self) -> str:
        """Get the deployment name."""
        return ''

    def endpoint_state(self) -> ResourceState:
        """Get the endpoint state."""
        if self._has_managed_identity():
            return self._get_endpoint_state_identity()
        else:
            return self._get_endpoint_state_token()

    def deployment_state(self) -> ResourceState:
        """Get the deployment state."""
        if self._has_managed_identity():
            return self._get_deployment_state_identity()
        else:
            return self._get_deployment_state_token()

    def create_endpoint(self) -> None:
        """Create the endpoint."""
        self._validate_model()
        endpoint = ManagedOnlineEndpoint(
            name=self.endpoint_name,
            description="this is a benchmark endpoint",
            auth_mode="key"
        )
        self._deploy_resources(endpoint)
        endpoint = self.ml_client.online_endpoints.get(name=self.endpoint_name)
        if self._online_endpoint_url is None:
            self._online_endpoint_url = endpoint.scoring_uri

    def create_deployment(self) -> None:
        """Create the deployment."""
        self._validate_model()
        deployment_env_vars = {
            "SUBSCRIPTION_ID": self._subscription_id,
            "RESOURCE_GROUP_NAME": self._resource_group,
            "WORKER_COUNT": 256
        }
        deployment_env_vars.update(self._additional_deployment_env_vars)
        max_concurrent_requests = 256
        deployment = ManagedOnlineDeployment(
            name=self.deployment_name,
            endpoint_name=self.endpoint_name,
            model=self._model.model_path,
            instance_type=self._sku,
            instance_count=1,
            code_configuration=None,
            environment=self._deployment_env,
            environment_variables=deployment_env_vars,
            request_settings=OnlineRequestSettings(
                request_timeout_ms=OSSOnlineEndpoint.REQUEST_TIMEOUT_MS,
                max_concurrent_requests_per_instance=max_concurrent_requests
            ),
            liveness_probe=ProbeSettings(
                failure_threshold=30,
                success_threshold=1,
                period=100,
                initial_delay=500,
            ),
            readiness_probe=ProbeSettings(
                failure_threshold=30,
                success_threshold=1,
                period=100,
                initial_delay=500,
            ),
        )
        self._deploy_resources(deployment)

    def delete_endpoint(self) -> None:
        """Delete the endpoint."""
        if self._has_managed_identity():
            self.ml_client.online_endpoints.begin_delete(name=self.endpoint_name).wait()
        else:
            LOGGER.warn("Delete endpoint is only supported when managed identity is enabled. Skipping now..")

    def delete_endpoint_model(self) -> None:
        """Delete the endpoint."""
        if self._has_managed_identity():
            self.ml_client.online_deployments.begin_delete(
                endpoint_name=self.endpoint_name, name=self.deployment_name).wait()
        else:
            LOGGER.warn("Delete deployment is only supported when managed identity is enabled. Skipping now..")

    def get_resource_authorization_header(self) -> dict:
        """Get the authorization header."""
        return {
            'Authorization': f'Bearer {self.curr_workspace._auth.get_authentication_header()}'
        }

    def get_endpoint_authorization_header(self) -> dict:
        """Get the authorization header."""
        if self._has_managed_identity():
            return self._get_endpoint_authorization_header_identity()
        return self._get_endpoint_authorization_header_token()

    def _get_endpoint_authorization_header_identity(self) -> dict:
        """Get the authorization header using managed identity."""
        return self._build_auth_headers(self.ml_client.online_endpoints.get_keys(self.endpoint_name).primary_key)

    def _get_endpoint_token(self) -> str:
        return self.ml_client.online_endpoints.get_keys(self.endpoint_name).primary_key

    def _get_endpoint_authorization_header_token(self) -> dict:
        """Get the authorization header using workspace authorization token."""
        resp = self._call_endpoint(get_requests_session().post, self._endpoint_list_key_url)
        self._raise_if_not_success(resp)
        content_dict = self._get_content_from_response(resp)
        self._build_auth_headers(content_dict['primaryKey'])

    def _build_auth_headers(self, token: str) -> dict:
        return {'Authorization': f'Bearer {token}'}

    def _deploy_resources(self, resource: Union[ManagedOnlineDeployment, ManagedOnlineEndpoint]) -> None:
        try:
            self.ml_client.begin_create_or_update(resource).wait()
        except Exception as err:
            raise BenchmarkValidationException._with_error(
                AzureMLError.create(
                    BenchmarkValidationError,
                    error_details=f"Deployment creation failed. Detailed Response:\n{err}."
                                  f" Please fix the issue and try to submit the job again.")
                )

    def _get_deployment_state_identity(self) -> ResourceState:
        return self._get_status_identity("deployment")

    def _get_deployment_state_token(self) -> ResourceState:
        # To use workspace tokens, we cannot retrieve the deployment state. So always return success.
        return ResourceState.SUCCESS

    def _get_endpoint_state_identity(self) -> ResourceState:
        """Get the endpoint state using managed identity."""
        return self._get_status_identity("endpoint")

    def _get_status_identity(self, resource_name: str) -> ResourceState:
        try:
            curr_state = "creating"
            while curr_state in ["creating", "updating", 'deleting', 'provisioning']:
                if resource_name == "endpoint":
                    resource = self.ml_client.online_endpoints.get(self.endpoint_name)
                    default_deployment_name = self._get_default_deployment_identity(resource)
                    if (self._generated_deployment_name or self._deployment_name is None) and default_deployment_name:
                        LOGGER.info("Using default deployment name %s", default_deployment_name)
                        self._deployment_name = default_deployment_name
                        self._generated_deployment_name = False
                else:
                    resource = self.ml_client.online_deployments.get(
                        endpoint_name=self.endpoint_name, name=self.deployment_name)
                curr_state = resource.provisioning_state.lower()
                if curr_state == "succeeded":
                    if resource_name == "endpoint" and self._online_endpoint_url is None:
                        self._online_endpoint_url = resource.scoring_uri
                    return ResourceState.SUCCESS
                elif curr_state == "failed":
                    return ResourceState.FAILURE
                time.sleep(30)
        except Exception as e:
            LOGGER.warn(e)
        return ResourceState.NOT_FOUND

    def _get_endpoint_state_token(self) -> ResourceState:
        """Get the endpoint state using workspace token."""
        resp = self._call_endpoint(
            get_requests_session().get, self._endpoint_url, self.get_resource_authorization_header())
        resource_state = self._get_resource_state(resp)
        LOGGER.info("Calling {} returned {} with content {}.".format(
            self._endpoint_url, resp.status_code, self._get_content_from_response(resp)))
        if resource_state == ResourceState.SUCCESS:
            content_dict = self._get_content_from_response(resp)
            if self._online_endpoint_url is None:
                self._online_endpoint_url = content_dict['endpoint']
            if content_dict['state'].lower() == "succeeded" and self.deployment_name in content_dict['trafficRules']:
                return ResourceState.SUCCESS
        raise BenchmarkValidationException._with_error(
            AzureMLError.create(
                BenchmarkValidationError,
                error_details='Endpoint {} does not exist. To use auto-deploy, '
                              'please assign managed identity to the compute.'.format(self.endpoint_name))
            )

    def _get_default_deployment_identity(self, endpoint: ManagedOnlineDeployment) -> str:
        """Get the default deployment name using managed identity."""
        return sorted(endpoint.traffic.items(), key=lambda x: -x[1])[0][0]

    @property
    def _endpoint_url(self) -> str:
        url_list = [
            _get_mms_url(self.curr_workspace), 'onlineEndpoints', self.endpoint_name
        ]
        return '/'.join(url_list)

    @property
    def _endpoint_list_key_url(self) -> str:
        url_list = [
            self._endpoint_url, 'listkeys'
        ]
        return '/'.join(url_list)

    @property
    def _deployment_url(self) -> str:
        url_list = [
            self._endpoint_url, 'deployments', self.deployment_name
        ]
        return '/'.join(url_list)

    def _has_managed_identity(self) -> bool:
        """Check if the compute has managed identity."""
        return self._credential is not None

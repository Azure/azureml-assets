# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Class for online endpoint."""


from typing import Any, Optional
from abc import abstractmethod
from enum import Enum
import json
import os
import uuid
from requests.models import Response
from azureml.core import Run, Workspace
from azure.identity import ManagedIdentityCredential
from azureml._restclient.clientbase import ClientBase
from azureml._common._error_definition.azureml_error import AzureMLError
from utils.error_definitions import BenchmarkValidationError
from utils.exceptions import BenchmarkValidationException
from .online_endpoint_model import OnlineEndpointModel

from ..logging import get_logger


LOGGER = get_logger(__name__)


class ResourceState(Enum):
    """Enum for resource state."""
    SUCCESS = "Success"
    FAILURE = "Failure"
    NOT_FOUND = "NotFound"


class OnlineEndpoint:
    """Class for AOAI and OSS online endpoint."""

    ENV_CLIENT_ID_KEY = "DEFAULT_IDENTITY_CLIENT_ID"
    SCOPE_AML = "https://ml.azure.com/.default"
    SCOPE_ARM = "https://management.azure.com/.default"

    def __init__(
            self,
            workspace_name: str,
            resource_group: str,
            subscription_id: str,
            online_endpoint_url: Optional[str] = None,
            endpoint_name: Optional[str] = None,
            deployment_name: Optional[str] = None,
            sku: Optional[str] = None,
            online_endpoint_model: Optional[OnlineEndpointModel] = None
    ):
        self._subscription_id = subscription_id
        self._resource_group = resource_group
        self._workspace_name = workspace_name
        self._endpoint_name = endpoint_name
        self._deployment_name = deployment_name
        self._online_endpoint_url = online_endpoint_url
        self._sku = sku
        self._model = online_endpoint_model
        self._endpoint_name = endpoint_name
        self._deployment_name = deployment_name
        self._generated_deployment_name = False
        if self._get_client_id() is None:
            LOGGER.info("Client id is not provided.")
            self._credential = None
        else:
            self._credential = self.get_credential()
        self._curr_worspace = None
        self._managed_identity_required = False

    @property
    def endpoint_name(self) -> str:
        """Get the endpoint name."""
        if self._endpoint_name is None:
            if self._online_endpoint_url is None:
                self._endpoint_name = ("b-" + str(uuid.uuid4().hex))[:16]
                LOGGER.info(f"Endpoint name is not provided, use a random one. {self._endpoint_name}")
            else:
                self._endpoint_name = self.get_endpoint_name_from_url()
                LOGGER.info(f"Endpoint name is not provided, use the one from url. {self._endpoint_name}")
        return self._endpoint_name
    
    @property
    def deployment_name(self) -> str:
        """Get the deployment name."""
        if self._deployment_name is None:
            if self._online_endpoint_url is None or self._model.is_oss_model():
                self._deployment_name = ("b-" + str(uuid.uuid4().hex))[:16]
                self._generated_deployment_name = True
                LOGGER.info(f"Deployment name is not provided, use a random one. {self._deployment_name}")
            else:
                self._deployment_name = self.get_deployment_name_from_url()
                LOGGER.info(f"Deployment name is not provided, use the one from url. {self._deployment_name}")
        return self._deployment_name

    @property
    def curr_workspace(self):
        """Get the current workspace."""
        if self._curr_worspace is None:
            self._curr_worspace = Run.get_context().experiment.workspace
        return self._curr_worspace
    
    @property
    def workspace_name(self) -> str:
        """Get the workspace name."""
        if self._workspace_name is None:
            self._workspace_name = self.curr_workspace.name
        return self._workspace_name
    
    @property
    def resource_group(self) -> str:
        """Get the resource group."""
        if self._resource_group is None:
            self._resource_group = self.curr_workspace.resource_group
            LOGGER.info(f"Resource group is not provided, use the one from workspace. {self._resource_group}")
        return self._resource_group
    
    @property
    def subscription_id(self) -> str:
        """Get the subscription id."""
        if self._subscription_id is None:
            self._subscription_id = self.curr_workspace.subscription_id
            LOGGER.info(f"Subscription id is not provided, use the one from workspace. {self._subscription_id}")
        return self._subscription_id

    @staticmethod
    def get_credential() -> ManagedIdentityCredential:
        """Get the credential."""
        client_id = os.environ.get(OnlineEndpoint.ENV_CLIENT_ID_KEY, None)
        credential = ManagedIdentityCredential(client_id=client_id)
        return credential
    
    @property
    def scoring_url(self) -> str:
        """Get the scoring url."""
        return self._online_endpoint_url
    
    @property
    def model(self) -> OnlineEndpointModel:
        """Get the model."""
        return self._model

    def _get_resource_state(self, resp: Response) -> ResourceState:
        """Get the resource state."""
        if resp.status_code == 200:
            return ResourceState.SUCCESS
        elif resp.status_code == 404:
            return ResourceState.NOT_FOUND
        else:
            return ResourceState.FAILURE

    def _get_resource_token(self) -> str:
        """Get the azure resource token."""
        token = self._credential.get_token(self._resource_scope)
        return token.token

    def _get_client_id(self) -> str:
        """Get the client id."""
        return os.environ.get(OnlineEndpoint.ENV_CLIENT_ID_KEY, None)

    def _get_content_from_response(self, resp: Response) -> dict:
        """Get the content from the response."""
        try:
            content = resp.content
            if isinstance(content, bytes):
                content = content.decode('utf-8')
            return json.loads(content)
        except Exception as err:
            LOGGER.error(f"Failed to get content from response: {err}")
            return {}

    @property    
    def _arm_base_url(self) -> str:
        """Get the arm base url."""
        url_list = [
            'https://management.azure.com', 'subscriptions', self.subscription_id,
            'resourceGroups', self.resource_group
        ]
        return "/".join(url_list)

    def _call_endpoint(
            self, call_method: Any, url: str, payload: Optional[dict]=None
    ) -> Response:
        headers = self.get_resource_authorization_header()
        resp = ClientBase._execute_func(
            call_method, url, params={}, headers=headers, json=payload
        )
        return resp

    @property
    def _resource_scope(self) -> str:
        return OnlineEndpoint.SCOPE_ARM

    @abstractmethod
    def get_endpoint_name_from_url(self) -> str:
        """Get the endpoint."""
        pass

    @abstractmethod
    def get_deployment_name_from_url(self) -> str:
        """Get the deployment name."""
        pass

    @abstractmethod
    def endpoint_state(self) -> ResourceState:
        """Check if the endpoint state."""
        pass

    @abstractmethod
    def deployment_state(self) -> ResourceState:
        """Check if the deployment state."""
        pass

    @abstractmethod
    def create_endpoint(self) -> None:
        """create the endpoint."""
        pass

    @abstractmethod
    def create_deployment(self) -> None:
        """create the deployment."""
        pass

    @abstractmethod
    def get_endpoint_authorization_header(self) -> dict:
        """Get the authorization header."""
        pass

    @abstractmethod
    def get_resource_authorization_header(self) -> dict:
        """Get the authorization header."""
        pass

    @abstractmethod
    def delete_endpoint(self):
        """Delete the endpoint."""
        pass

    @abstractmethod
    def delete_deployment(self):
        """Delete the deployment."""
        pass

    def _raise_if_not_success(self, resp: Response, msg: Optional[str]=None) -> None:
        """Raise error if not success."""
        default_msg = f'Failed to due to {resp.content} after getting {resp.status_code}'
        if resp.status_code not in (200, 201, 202):
            raise BenchmarkValidationException._with_error(
                AzureMLError.create(
                    BenchmarkValidationError,
                    error_details=msg if msg else default_msg)
                )

    def _validate_settings(self) -> None:
        """Validate the settings."""
        if self._managed_identity_required:
            self._validate_managed_identity()

    def _validate_managed_identity(self) -> None:
        if self._credential is None:
            raise BenchmarkValidationException._with_error(
                AzureMLError.create(
                    BenchmarkValidationError,
                    error_details='Managed identity is required for this scenario.')
                )

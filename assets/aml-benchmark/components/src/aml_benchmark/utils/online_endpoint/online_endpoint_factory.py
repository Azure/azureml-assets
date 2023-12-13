# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Factory for online endpoint."""


from typing import Optional
from .aoai_online_endpoint import AOAIOnlineEndpoint
from .online_endpoint import OnlineEndpoint
from .online_endpoint_model import OnlineEndpointModel
from .oss_online_endpoint import OSSOnlineEndpoint


class OnlineEndpointFactory:
    """Factory for online endpoint."""

    @staticmethod
    def get_online_endpoint(
            workspace_name: str,
            resource_group: str,
            subscription_id: str,
            online_endpoint_model: OnlineEndpointModel,
            endpoint: Optional[str] = None,
            deployment_name: Optional[str] = None,
            sku: Optional[str] = None,
            location: Optional[str] = None,
            connections_name: Optional[str] = None,
            additional_deployment_env_vars: dict = {},
            deployment_env: str = None
    ) -> OnlineEndpoint:
        """Get the online endpoint."""
        online_endpoint_url = endpoint if OnlineEndpointFactory._is_endpoint_url(endpoint) else None
        endpoint_name = endpoint if online_endpoint_url is None else None
        if online_endpoint_model.is_aoai_model():
            return AOAIOnlineEndpoint(
                workspace_name,
                resource_group,
                subscription_id,
                online_endpoint_model,
                online_endpoint_url,
                endpoint_name,
                deployment_name,
                sku,
                location,
                connections_name=connections_name
            )
        else:
            return OSSOnlineEndpoint(
                workspace_name,
                resource_group,
                subscription_id,
                online_endpoint_model,
                online_endpoint_url,
                endpoint_name,
                deployment_name,
                sku,
                connections_name=connections_name,
                additional_deployment_env_vars=additional_deployment_env_vars,
                deployment_env=deployment_env
            )

    @staticmethod
    def from_metadata(
        metadata_dict: dict
    ) -> OnlineEndpoint:
        """Get the online endpoint from metadata dict."""
        online_endpoint_model = OnlineEndpointModel(
            metadata_dict['model_path'],
            None,
            metadata_dict['model_type']
        )
        return OnlineEndpointFactory.get_online_endpoint(
            workspace_name=metadata_dict['workspace_name'],
            resource_group=metadata_dict['resource_group'],
            subscription_id=metadata_dict['subscription_id'],
            online_endpoint_model=online_endpoint_model,
            endpoint=metadata_dict['endpoint_name'],
            deployment_name=metadata_dict['deployment_name'],
            sku=None,
            location=None,
            connections_name=metadata_dict['connection_name']
        )

    @staticmethod
    def _is_endpoint_url(endpoint: str) -> bool:
        """Check if the endpoint is url."""
        return endpoint is not None and endpoint.startswith('http')

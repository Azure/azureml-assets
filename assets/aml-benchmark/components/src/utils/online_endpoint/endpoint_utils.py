# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Class for endpoint utilities."""


import os
import json

from .online_endpoint import OnlineEndpoint


class EndpointUtilities:
    """Class for endpoint utilities."""

    METADATA_FILE = "endpoint_metadata.json"
    DELETE_STATUS_FILE = "delete_status.json"

    @staticmethod
    def dump_endpoint_metadata_json(
            online_endpoint: OnlineEndpoint,
            is_managed_endpoint: bool,
            is_managed_deployment: bool,
            is_managed_connections: bool,
            output_dir: str
    ):
        """Dump endpoint info metadata json."""
        headers = {'Content-Type': 'application/json'}
        if online_endpoint.model.is_oss_model():
            headers["azureml-model-deployment"] = online_endpoint.deployment_name
        endpoint_metadata = {
            "endpoint_name": online_endpoint.endpoint_name,
            "scoring_url": online_endpoint.scoring_url,
            "deployment_name": online_endpoint.deployment_name,
            "is_managed_endpoint": is_managed_endpoint,
            "is_managed_deployment": is_managed_deployment,
            "is_managed_connections": is_managed_connections,
            "scoring_headers": headers,
            "workspace_name": online_endpoint.workspace_name,
            "resource_group": online_endpoint.resource_group,
            "subscription_id": online_endpoint.subscription_id,
            "model_path": online_endpoint.model.model_path,
            "model_type": online_endpoint.model.model_type,
            "connections_name": online_endpoint.connections_name,
        }
        with open(os.path.join(output_dir, EndpointUtilities.METADATA_FILE), 'w') as metadata_file:
            json.dump(endpoint_metadata, metadata_file)

    @staticmethod
    def load_endpoint_metadata_json(output_dir: str) -> dict:
        """Load endpoint info metadata json."""
        with open(os.path.join(output_dir, EndpointUtilities.METADATA_FILE), 'r') as metadata_file:
            return json.load(metadata_file)
        
    @staticmethod
    def dump_delete_status(
        is_endpoint_deleted: bool,
        is_deployment_deleted: bool,
        is_connections_deleted: bool,
        output_dir: str
    ) -> None:
        """Dump delete status."""
        delete_status = {
            "is_endpoint_deleted": is_endpoint_deleted,
            "is_deployment_deleted": is_deployment_deleted,
            "is_connections_deleted": is_connections_deleted,
        }
        with open(os.path.join(output_dir, EndpointUtilities.DELETE_STATUS_FILE), 'w') as delete_status_file:
            json.dump(delete_status, delete_status_file)

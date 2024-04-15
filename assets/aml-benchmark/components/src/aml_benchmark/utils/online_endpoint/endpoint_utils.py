# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Class for endpoint utilities."""
import os
from typing import Any
import json
import hashlib
import re

from .online_endpoint import OnlineEndpoint
from .online_endpoint_model import OnlineEndpointModel
from ..logging import get_logger


logger = get_logger(__name__)


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
            output_path: str
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
            "connection_name": online_endpoint.connections_name
        }
        # if output_path is a directory, append the file name
        if os.path.isdir(output_path):
            output_path = os.path.join(output_path, EndpointUtilities.METADATA_FILE)
        with open(output_path, 'w') as metadata_file:
            json.dump(endpoint_metadata, metadata_file)

    @staticmethod
    def load_endpoint_metadata_json(output_path: str) -> dict:
        """Load endpoint info metadata json."""
        if os.path.isdir(output_path):
            output_path = os.path.join(output_path, EndpointUtilities.METADATA_FILE)
            # verify the file exists
            if not os.path.exists(output_path):
                raise FileNotFoundError(f"Endpoint metadata file {output_path} not found.")
        with open(output_path, 'r') as metadata_file:
            return json.load(metadata_file)

    @staticmethod
    def dump_delete_status(
        is_endpoint_deleted: bool,
        is_deployment_deleted: bool,
        is_connections_deleted: bool,
        output_path: str
    ) -> None:
        """Dump delete status."""
        delete_status = {
            "is_endpoint_deleted": is_endpoint_deleted,
            "is_deployment_deleted": is_deployment_deleted,
            "is_connections_deleted": is_connections_deleted,
        }
        # if output_path is a directory, append the file name
        if os.path.isdir(output_path):
            output_path = os.path.join(output_path, EndpointUtilities.DELETE_STATUS_FILE)
        with open(output_path, 'w') as delete_status_file:
            json.dump(delete_status, delete_status_file)

    @staticmethod
    def _hash_string_character(input_string: str) -> str:
        """Hash a string only with its alphanumerical characters."""
        return hashlib.sha256(
            re.sub(r'[^A-Za-z0-9 ]+', '', input_string).encode('utf-8')).hexdigest()

    @staticmethod
    def hash_payload_prompt(
        payload: Any, model: OnlineEndpointModel
    ) -> str:
        """Hash the payload and prompt."""
        try:
            if model.is_oss_model() or model.is_aoai_model():
                if "input_data" in payload and "input_string" in payload["input_data"]:
                    # For OSS endpoint
                    prompt = payload["input_data"]["input_string"]
                    if isinstance(prompt[0], dict):
                        # For OSS chat model
                        prompt = " ".join([p["content"] for p in prompt])
                    elif isinstance(prompt[0], str):
                        # For OSS text generation model.
                        prompt = " ".join(prompt)
                elif "messages" in payload:
                    # For OAI chat model
                    if isinstance(payload["messages"], list):
                        prompt = " ".join([msg.get("content", "") for msg in payload["messages"]])
                    else:
                        # unknown format, try to stringify all of them.
                        prompt = str(payload["messages"])
                elif "prompt" in payload:
                    # For OAI text generation model
                    prompt = payload["prompt"]
                else:
                    # unknown format, try to stringify all of them.
                    prompt = str(payload)
            elif model.is_claude_model():
                prompt = str(payload["prompt"])
            else:
                # if model is unknown, use the full payload.
                prompt = str(payload)
        except Exception as e:
            logger.warning(
                "Failed to get the prompt from payload {} use the full payload now.".format(e))
            prompt = str(payload)
        return EndpointUtilities._hash_string_character(str(prompt))

    @staticmethod
    def get_model_from_deployment_config_file(dir_path: str) -> OnlineEndpointModel:
        """Get the online endpoint model from deployment config file."""
        config = EndpointUtilities.load_endpoint_metadata_json(dir_path)
        return OnlineEndpointModel(
            config['model_path'],
            None,
            config['model_type'],
            config['scoring_url']
        )

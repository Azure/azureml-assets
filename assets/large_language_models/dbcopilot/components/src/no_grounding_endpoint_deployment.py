# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""The component deploys the endpoint for db copilot."""
import json
import logging
import os
import shutil
import tempfile
from typing import Dict

from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Model
from component_base import main_entry_point
from endpoint_deployment_base import EndpointDeploymentBase


@main_entry_point("deploy")
class NoGroundingEndpointDeployment(EndpointDeploymentBase):
    """EndpointDeployment Class."""

    parameter_type_mapping: Dict[str, str] = {
        "cache_path": "uri_folder",
    }

    parameter_mode_mapping: Dict[str, str] = {
        "cache_path": "direct",
    }

    def __init__(self) -> None:
        """Initialize the class."""
        super().__init__()

    def deploy(
        self,
        deployment_name: str,
        endpoint_name: str,
        embedding_connection: str,
        embedding_aoai_deployment_name: str,
        chat_connection: str,
        chat_aoai_deployment_name: str,
        mir_environment: str,
        configs_json: str = None,
        cache_path: str = None,
    ):
        """deploy_endpoint."""
        from utils.asset_utils import get_full_env_path

        model = None
        extra_environment = {}

        if cache_path:
            model = Model(
                name=f"endpoint-{endpoint_name}-{deployment_name}-cache",
                path=cache_path,
                type=AssetTypes.CUSTOM_MODEL,
                description=f"cache for endpoint {endpoint_name} deployment: {deployment_name}",
            )
            extra_environment["DBCOPILOT_CACHE_URI"] = cache_path
        os.environ[
            "AZUREML_WORKSPACE_CONNECTION_ID_AOAI_EMBEDDING"
        ] = embedding_connection
        os.environ["AZUREML_WORKSPACE_CONNECTION_ID_AOAI_CHAT"] = chat_connection
        secrets_dict = self.get_secrets()
        secrets_dict["embedding-deploy-name"] = embedding_aoai_deployment_name
        secrets_dict["chat-deploy-name"] = chat_aoai_deployment_name
        mir_environment = get_full_env_path(self.mlclient_credential, mir_environment)
        with tempfile.TemporaryDirectory() as code_dir:
            logging.info("code_dir: %s", code_dir)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            shutil.copytree(
                os.path.join(current_dir, "db_copilot_mir/code"),
                code_dir,
                dirs_exist_ok=True,
            )
            with open(os.path.join(code_dir, "secrets.json"), "w") as f:
                json.dump(secrets_dict, f)
            logging.info("dumped secrets to secrets.json")
            if configs_json is not None:
                configs = json.loads(configs_json)
                assert isinstance(configs, list), "configs_json must be a JSON list"
                with open(os.path.join(code_dir, "configs.json"), "w") as f:
                    json.dump(configs, f)
            self._deploy_endpoint(
                mir_environment,
                endpoint_name,
                deployment_name,
                code_dir,
                "score_zero.py",
                model=model,
                extra_environment_variables=extra_environment,
            )

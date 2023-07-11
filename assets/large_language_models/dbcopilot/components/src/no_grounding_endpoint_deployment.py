# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""The component deploys the endpoint for db copilot."""
import json
import logging
import os
import shutil
import tempfile

from component_base import main_entry_point
from endpoint_deployment_base import EndpointDeploymentBase


@main_entry_point("deploy")
class NoGroundingEndpointDeployment(EndpointDeploymentBase):
    """EndpointDeployment Class."""

    def __init__(self) -> None:
        """Initialize the class."""
        super().__init__()

    def deploy(
        self,
        deployment_name: str,
        endpoint_name: str,
        embedding_aoai_deployment_name: str,
        chat_aoai_deployment_name: str,
        mir_environment: str,
    ):
        """deploy_endpoint."""
        from utils.asset_utils import get_full_env_path

        secrets_dict = self.get_secrets()
        secrets_dict["embedding-deploy-name"] = embedding_aoai_deployment_name
        secrets_dict["chat-deploy-name"] = chat_aoai_deployment_name
        mir_environment = get_full_env_path(self.mlclient_credential, mir_environment)
        with tempfile.TemporaryDirectory() as code_dir:
            logging.info("code_dir: %s", code_dir)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            shutil.copytree(os.path.join(current_dir, "db_copilot_mir/code"), code_dir, dirs_exist_ok=True)
            with open(os.path.join(code_dir, "secrets.json"), "w") as f:
                json.dump(secrets_dict, f)
            logging.info("dumped secrets to secrets.json")
            self._deploy_endpoint(mir_environment, endpoint_name, deployment_name, code_dir, "score_zero.py")

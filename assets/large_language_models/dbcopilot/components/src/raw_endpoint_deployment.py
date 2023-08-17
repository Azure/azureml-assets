# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""The component deploys the endpoint for db copilot."""
import json
import logging
import os
import shutil
import tempfile
from typing import Dict

from component_base import main_entry_point
from endpoint_deployment_base import EndpointDeploymentBase


@main_entry_point("deploy")
class RawEndpointDeployment(EndpointDeploymentBase):
    """EndpointDeployment Class."""

    def __init__(self) -> None:
        """Initialize the class."""
        super().__init__()

    parameter_type_mapping: Dict[str, str] = {"code_uri": "uri_folder"}

    def deploy(
        self,
        deployment_name: str,
        endpoint_name: str,
        mir_environment: str,
        code_folder: str,
        app_module_name: str = "app",
        app_name: str = "app",
    ):
        """deploy_endpoint."""
        from utils.asset_utils import get_full_env_path

        mir_environment = get_full_env_path(self.mlclient_credential, mir_environment)
        with tempfile.TemporaryDirectory() as code_dir:
            logging.info("code_dir: %s", code_dir)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            score_code_dir = os.path.join(current_dir, "db_copilot_mir/code")
            logging.info("copying code from %s to %s", score_code_dir, code_dir)
            shutil.copytree(score_code_dir, code_dir, dirs_exist_ok=True)
            logging.info(
                "copying code from %s to %s",
                code_folder,
                os.path.join(code_dir, "app_code"),
            )
            shutil.copytree(
                code_folder, os.path.join(code_dir, "app_code"), dirs_exist_ok=True
            )
            with open(os.path.join(code_dir, "app_config.json"), "w") as f:
                app_config = {"module_name": app_module_name, "app_name": app_name}
                json.dump(app_config, f)
            logging.info("Starting deploy endpoint")
            extra_environment_variables = {
                "RSLEX-MOUNT": f"azureml://subscriptions/{self.workspace.subscription_id}/resourcegroups/{self.workspace.resource_group}/workspaces/{self.workspace.name}/datastores/workspaceblobstore"  # noqa: E501
            }
            self._deploy_endpoint(
                mir_environment,
                endpoint_name,
                deployment_name,
                code_dir,
                "general_score.py",
                extra_environment_variables=extra_environment_variables,
            )

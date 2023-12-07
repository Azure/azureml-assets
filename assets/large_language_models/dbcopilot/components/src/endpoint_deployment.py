# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""The component deploys the endpoint for db copilot."""
import json
import logging
import os
import shutil
import tempfile
from dataclasses import asdict
from typing import Dict

from component_base import main_entry_point
from db_copilot_tool.contracts.db_copilot_config import DBCopilotConfig
from endpoint_deployment_base import EndpointDeploymentBase


@main_entry_point("deploy")
class EndpointDeployment(EndpointDeploymentBase):
    """EndpointDeployment Class."""

    def __init__(self) -> None:
        """Initialize the class."""
        super().__init__()

    parameter_type_mapping: Dict[str, str] = {
        "grounding_embedding_uri": "uri_folder",
        "example_embedding_uri": "uri_folder",
        "db_context_uri": "uri_folder",
    }

    parameter_mode_mapping: Dict[str, str] = {
        "db_context_uri": "direct",
        "grounding_embedding_uri": "direct",
        "example_embedding_uri": "direct",
    }

    def deploy(
        self,
        deployment_name: str,
        endpoint_name: str,
        grounding_embedding_uri: str,
        embedding_aoai_deployment_name: str,
        chat_aoai_deployment_name: str,
        db_context_uri: str,
        asset_uri: str,
        mir_environment: str,
        example_embedding_uri: str = None,
        selected_tables: str = None,
        max_tables: int = None,
        max_columns: int = None,
        max_rows: int = None,
        max_text_length: int = None,
        tools: str = None,
        temperature: float = 0.0,
        top_p: float = 0.0,
        knowledge_pieces: str = None,
        sku: str = "Standard_DS3_v2",
    ):
        """deploy_endpoint."""
        from utils.asset_utils import get_datastore_uri

        workspace = self.workspace
        # find datastore uri

        datastore_uri = get_datastore_uri(workspace, asset_uri)
        logging.info(f"Datastore uri: {datastore_uri}")
        secrets_dict = self.get_secrets()

        config = DBCopilotConfig(
            grounding_embedding_uri=grounding_embedding_uri,
            example_embedding_uri=example_embedding_uri,
            db_context_uri=db_context_uri,
            chat_aoai_deployment_name=chat_aoai_deployment_name,
            datastore_uri=datastore_uri,
            embedding_aoai_deployment_name=embedding_aoai_deployment_name,
            history_cache_enabled=True,
            selected_tables=selected_tables,
            max_tables=max_tables,
            max_columns=max_columns,
            max_rows=max_rows,
            max_text_length=max_text_length,
            tools=tools,
            temperature=temperature,
            top_p=top_p,
            knowledge_pieces=knowledge_pieces,
        )
        logging.info(f"DBCopilotConfig: {config}")
        # mir_environment = get_full_env_path(self.mlclient_credential, mir_environment)
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
            with open(os.path.join(code_dir, "configs.json"), "w") as f:
                json.dump([asdict(config)], f)
            self._deploy_endpoint(
                mir_environment,
                endpoint_name,
                deployment_name,
                code_dir,
                score_script="score_zero.py",
                sku=sku,
            )

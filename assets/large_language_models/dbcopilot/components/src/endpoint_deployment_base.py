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

from azure.ai.ml.entities import (
    CodeConfiguration,
    Environment,
    IdentityConfiguration,
    ManagedIdentityConfiguration,
    ManagedOnlineDeployment,
    ManagedOnlineEndpoint,
    OnlineRequestSettings,
)
from azureml.rag.utils.connections import (
    get_connection_by_id_v2,
    workspace_connection_to_credential,
)
from component_base import OBOComponentBase, main_entry_point
from db_copilot_tool.contracts.db_copilot_config import DBCopilotConfig


@main_entry_point("deploy")
class EndpointDeploymentBase(OBOComponentBase):
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

    def get_secrets(self):
        """Get aoai secrets."""
        embedding_connection_id = os.environ.get(
            "AZUREML_WORKSPACE_CONNECTION_ID_AOAI_EMBEDDING", None
        )
        chat_connection_id = os.environ.get(
            "AZUREML_WORKSPACE_CONNECTION_ID_AOAI_CHAT", None
        )
        secrets = {}
        if (
            embedding_connection_id is not None
            and embedding_connection_id != ""
            and chat_connection_id is not None
            and chat_connection_id != ""
        ):
            for connection_type, connection_id in {
                "embedding": embedding_connection_id,
                "chat": chat_connection_id,
            }.items():
                connection = get_connection_by_id_v2(connection_id)
                credential = workspace_connection_to_credential(connection)
                if hasattr(credential, "key"):
                    secrets.update(
                        {
                            f"{connection_type}-aoai-api-key": credential.key,
                            f"{connection_type}-aoai-api-base": connection[
                                "properties"
                            ].get("target", {}),
                        }
                    )
                    logging.info("Using workspace connection key for OpenAI")
        else:
            raise ValueError(
                "Please specify the connection id (AZUREML_WORKSPACE_CONNECTION_ID_AOAI_EMBEDDING & AZUREML_WORKSPACE_CONNECTION_ID_AOAI_CHAT) for embedding and chat"  # noqa: E501
            )

        return secrets

    def _deploy_endpoint(
        self,
        mir_environment: str,
        endpoint_name: str,
        deployment_name: str,
        code_dir: str,
        score_script: str = "score.py",
        extra_environment_variables: Dict[str, str] = None,
    ):
        from utils.asset_utils import get_full_env_path

        endpoint = ManagedOnlineEndpoint(
            name=endpoint_name,
            description="DB CoPilot MIR endpoint",
            auth_mode="key",
            identity=IdentityConfiguration(
                type="UserAssigned",
                user_assigned_identities=[ManagedIdentityConfiguration(client_id=None)],
            ),
        )
        mir_environment = get_full_env_path(self.mlclient_credential, mir_environment)
        env = (
            mir_environment
            if mir_environment.startswith("azureml://")
            else Environment(image=mir_environment)
        )
        logging.info("Environment: %s", env)
        environment_variables = {
            "EMBEDDING_STORE_LOCAL_CACHE_PATH": "/tmp/embedding_store_cache",
            "WORKER_COUNT": "1",
            "WORKER_TIMEOUT": "0",
            "GUNICORN_CMD_ARGS": "--threads 20",
            "AML_CORS_ORIGINS": "*",
        }
        if extra_environment_variables:
            environment_variables.update(extra_environment_variables)
        deployment = ManagedOnlineDeployment(
            name=deployment_name,
            endpoint_name=endpoint_name,
            environment=env,
            environment_variables=environment_variables,
            code_configuration=CodeConfiguration(
                code=code_dir, scoring_script=score_script
            ),
            instance_type="Standard_DS3_v2",
            instance_count=1,
            request_settings=OnlineRequestSettings(
                request_timeout_ms=90000,
                max_concurrent_requests_per_instance=1000,
                max_queue_wait_ms=90000,
            ),
        )
        logging.info("Deployment: %s", deployment)
        ml_client = self.ml_client
        try:
            ml_client.online_endpoints.get(endpoint_name)
        except Exception:
            logging.info(f"Creating endpoint {endpoint_name}")
            endpoint_poller = ml_client.online_endpoints.begin_create_or_update(
                endpoint, local=False
            )
            endpoint_poller.result()
            logging.info(f"Created endpoint {endpoint_name}")
        deployment_poller = ml_client.online_deployments.begin_create_or_update(
            deployment=deployment, local=False
        )
        deployment_result = deployment_poller.result()
        logging.info(
            f"Created deployment {deployment_name}. Result: {deployment_result}"
        )

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
    ):
        """deploy_endpoint."""
        from utils.asset_utils import get_datastore_uri

        workspace = self.workspace
        # find datastore uri

        datastore_uri = get_datastore_uri(workspace, asset_uri)
        logging.info(f"Datastore uri: {datastore_uri}")
        secrets_dict = self.get_secrets()
        if selected_tables:
            selected_tables = json.loads(selected_tables)
            if not isinstance(selected_tables, list):
                raise ValueError("selected_tables must be a list")

        if tools:
            tools = json.loads(tools)
            if not isinstance(tools, list):
                raise ValueError("tools must be a dict")

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
            with open(os.path.join(code_dir, "db_copilot_config.json"), "w") as f:
                json.dump(asdict(config), f)
            self._deploy_endpoint(
                mir_environment, endpoint_name, deployment_name, code_dir
            )

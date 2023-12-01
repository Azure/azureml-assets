# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""The component deploys the endpoint for db copilot."""
import logging
import os
from typing import Dict, Optional

from azure.ai.ml.entities import (
    CodeConfiguration,
    Environment,
    IdentityConfiguration,
    ManagedIdentityConfiguration,
    ManagedOnlineDeployment,
    ManagedOnlineEndpoint,
    Model,
    OnlineRequestSettings,
)
from azureml.rag.utils.connections import (
    connection_to_credential,
    get_connection_by_id_v2,
)
from component_base import OBOComponentBase, main_entry_point


@main_entry_point("deploy")
class EndpointDeploymentBase(OBOComponentBase):
    """EndpointDeployment Class."""

    def __init__(self) -> None:
        """Initialize the class."""
        super().__init__()

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
                connection = get_connection_by_id_v2(
                    connection_id, credential=self.mlclient_credential
                )
                credential = connection_to_credential(connection)
                if hasattr(credential, "key"):
                    secrets.update(
                        {
                            f"{connection_type}-aoai-api-key": credential.key,
                            f"{connection_type}-aoai-api-base": connection.target,
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
        model: Optional[Model] = None,
        sku: str = "Standard_DS3_v2",
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
            model=model,
            code_configuration=CodeConfiguration(
                code=code_dir, scoring_script=score_script
            ),
            instance_type=sku,
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
        try:
            deployment_poller = ml_client.online_deployments.begin_create_or_update(
                deployment=deployment, local=False
            )
            deployment_result = deployment_poller.result()
            logging.info(
                f"Created deployment {deployment_name}. Result: {deployment_result}"
            )
        except Exception as e:
            logging.error(f"Deployment failed: {e}")
            logs = ml_client.online_deployments.get_logs(
                name=deployment_name, endpoint_name=endpoint_name, lines=100
            )
            logging.error(f"Endpoint deployment logs: {logs}")
            raise e

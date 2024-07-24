# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Data generator utils."""

import os
import time

from abc import ABC, abstractmethod
from azure.ai.ml import MLClient
from azure.ai.ml.identity import AzureMLOnBehalfOfCredential
from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment, ServerlessEndpoint
from azure.identity import AzureCliCredential, ManagedIdentityCredential
from azureml.acft.common_components import get_logger_app
from azureml.core import Run, Workspace
from azureml.core.run import _OfflineRun

from typing import List, Union


logger = get_logger_app("azureml.acft.contrib.hf.nlp.entry_point.data_import.data_import")
RETRY_DELAY = 5


def retry(times: int):
    """Retry utility to wrap.

    Args:
        times (int): No of retries
    """

    def decorator(func):
        def newfn(*args, **kwargs):
            attempt = 1
            while attempt <= times:
                try:
                    return func(*args, **kwargs)
                except Exception:
                    attempt += 1
                    ex_msg = "Exception thrown when attempting to run {}, attempt {} of {}".format(
                        func.__name__, attempt, times
                    )
                    logger.warning(ex_msg)
                    if attempt < times:
                        time.sleep(RETRY_DELAY)
                    else:
                        logger.warning(
                            "Retried {} times when calling {}, now giving up!".format(times, func.__name__)
                        )
                        raise

        return newfn

    return decorator


def get_credential() -> Union[ManagedIdentityCredential, AzureMLOnBehalfOfCredential]:
    """Create and validate credentials."""
    # try msi, followed by obo, followed by azure cli
    credential = None
    try:
        msi_client_id = os.environ.get("DEFAULT_IDENTITY_CLIENT_ID")
        credential = ManagedIdentityCredential(client_id=msi_client_id)
        credential.get_token("https://management.azure.com/.default")
        logger.info("Using MSI creds")
        return credential
    except Exception:
        logger.error("MSI auth failed")
    try:
        credential = AzureMLOnBehalfOfCredential()
        credential.get_token("https://management.azure.com/.default")
        logger.info("Using OBO creds")
        return credential
    except Exception:
        logger.error("OBO cred failed")
    try:
        credential = AzureCliCredential()
        credential.get_token("https://management.azure.com/.default")
        logger.info("Using OBO creds")
        return credential
    except Exception:
        logger.error("Azure CLI cred failed")

    raise Exception("Error creating credentials.")


def get_workspace() -> Workspace:
    """Return current workspace."""
    run = Run.get_context()
    if isinstance(run, _OfflineRun):
        ws: Workspace = Workspace.from_config("config.json")
    else:
        ws: Workspace = run.experiment.workspace
    return ws


def get_workspace_mlclient(workspace: Workspace = None) -> MLClient:
    """Return workspace mlclient."""
    credential = get_credential()
    workspace = get_workspace() if workspace is None else workspace
    if credential and workspace:
        return MLClient(
            credential,
            subscription_id=workspace.subscription_id,
            resource_group_name=workspace.resource_group,
            workspace_name=workspace.name
        )
    raise Exception("Error creating MLClient. No credentials or workspace found")


class EndpointDetails(ABC):
    """Base class for endpoint details."""

    @abstractmethod
    def get_endpoint_key(self) -> str:
        """Get endpoint key."""
        raise NotImplementedError()

    @abstractmethod
    def get_endpoint_url(self) -> str:
        """Get endpoint URL."""
        raise NotImplementedError()

    def get_deployed_model_id(self) -> str:
        """Get deployment model asset id."""
        raise NotImplementedError()


class ServerlessEndpointDetails(EndpointDetails):
    """Serverless endpoint details for data generation."""

    def __init__(self, mlclient_ws: MLClient, endpoint_name: str):
        """Initialize endpoint details for serverless endpoint.

        Args:
            mlclient_ws (MLClient): workspace mlclient
            endpoint_name (str): managed online endpoint name

        Raises:
            Exception: if fetching endpoint details fail
            Exception: if endpoint is not in healthy state
        """
        self._mlclient: MLClient = mlclient_ws
        try:
            self._endpoint: ServerlessEndpoint = self._mlclient.serverless_endpoints.get(endpoint_name)
        except Exception as e:
            raise Exception(f"Serverless endpoint fetch details failed with exception: {e}")

        # ensure endpoint is healthy
        logger.info(f"Endpoint provisioning state: {self._endpoint.provisioning_state}")
        if not (self._endpoint.provisioning_state.lower() == "succeeded"):
            raise Exception(f"Endpoint {self._endpoint.name} is unhealthy.")

    def get_endpoint_key(self) -> str:
        """Get endpoint primary key for serverless deployment.

        Raises:
            Exception: if fetching key fails

        Returns:
            str: endpoint primary key for serverless deployment.
        """
        try:
            return self._mlclient.serverless_endpoints.get_keys(self._endpoint.name).primary_key
        except Exception as e:
            raise Exception(f"Failed to get endpoint keys for endpoint: {self._endpoint.name}. Exception: {e}")

    def get_endpoint_url(self) -> str:
        """Get URL for managed online endpoint."""
        return self._endpoint.scoring_uri

    def get_deployed_model_id(self) -> str:
        """Return deployed model id."""
        return self._endpoint.model_id


class OnlineEndpointDetails(EndpointDetails):
    """Online endpoint details for data generation."""
    
    def __init__(self, mlclient_ws: MLClient, endpoint_name: str):
        """Initialize endpoint details for managed online endpoint.

        Args:
            mlclient_ws (MLClient): workspace mlclient
            endpoint_name (str): managed online endpoint name

        Raises:
            Exception: if fetching endpoint details fail
            Exception: if traffic is not allocated or allocated to more than one endpoint
            Exception: if endpoint or deployment is not in healthy state
        """
        self._mlclient: MLClient = mlclient_ws
        try:
            self._endpoint: ManagedOnlineEndpoint = self._mlclient.online_endpoints.get(endpoint_name)
        except Exception as e:
            raise Exception(f"Online endpoint fetch details failed with exception: {e}")

        all_deployments = self._get_deployments()
        # fetch deployment with 100% traffic 
        deployments = [
            deployment
            for deployment in all_deployments if deployment.name in [
                deployment_name 
                for deployment_name, traffic_percent in self._endpoint.traffic.items() if traffic_percent == 100
            ]
        ]

        if len(deployments) != 1:
            raise Exception(
                f"Endpoint {self._endpoint.name} does not meet traffic criteria "
                "of one alone deployment with 100% traffic allocation. "
                f"Currently trafic is allocated to {len(self._endpoint.traffic)} deployments"
            )

        self._deployment = deployments[0]
        # ensure endpoint and deployment is healthy
        logger.info(f"Endpoint provisioning state: {self._endpoint.provisioning_state}")
        logger.info(f"Deployment provisioning state: {self._deployment.provisioning_state}")
        if not (self._endpoint.provisioning_state.lower() == "succeeded" \
                and self._deployment.provisioning_state.lower() == "succeeded"):
            raise Exception(f"Endpoint {self._endpoint.name} or deployment {self._deployment.name} is unhealthy.")

    def get_endpoint_key(self):
        """Get endpoint primary key for managed online deployment.

        Raises:
            Exception: if fetching key fails

        Returns:
            str: endpoint primary key for managed online deployment.
        """
        try:
            return self._mlclient.online_endpoints.get_keys(self._endpoint.name).primary_key
        except Exception as e:
            raise Exception(f"Failed to get endpoint keys for endpoint: {self._endpoint.name}. Exception: {e}")

    def get_endpoint_url(self) -> str:
        """Get URL for managed online endpoint."""
        return self._endpoint.scoring_uri

    def get_deployed_model_id(self):
        """Return deployed model id."""
        return self._deployment.model

    def _get_deployments(self) -> List[ManagedOnlineDeployment]:
        """Return list of deployment for current endpoint.

        Returns:
            List[ManagedOnlineDeployment]: List of deployments
        """
        try:
            self._deployments: List[ManagedOnlineDeployment] = self._mlclient.online_deployments.list(self._endpoint.name)
            return self._deployments
        except Exception as e:
            logger.error(f"Could not fetch deployments for endpoint: {self._endpoint.name}. Exception => {e}")
            return None


def get_endpoint_details(mlclient_ws: MLClient, endpoint_name: str) -> EndpointDetails:
    """Get endpoint details for endpoint created in workspace or ai project.

    Args:
        mlclient_ws (MLClient): workspace mlclient
        endpoint_name (str): endpoint name of the workspace/aiproject endpoint

    Raises:
        Exception: if endpoint details are not found

    Returns:
        EndpointDetails: endpoint details for endpoint name
    """
    logger.info("Checking if endpoint is a serverless deployment")
    try:
        return ServerlessEndpointDetails(mlclient_ws, endpoint_name)
    except Exception as e:
        logger.warning(f"Fetching serverless endpoint details failed with => {e}")

    try:
        return OnlineEndpointDetails(mlclient_ws, endpoint_name)
    except Exception as e:
        logger.warning(f"Fetching serverless endpoint details failed with => {e}")

    raise Exception(
        f"Could not fetch endpoint {endpoint_name} details for online or serverless deployment."
    )

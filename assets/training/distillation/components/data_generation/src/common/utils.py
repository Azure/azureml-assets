# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Data generator utils."""

import os
import time

from azure.ai.ml import MLClient
from azure.ai.ml.identity import AzureMLOnBehalfOfCredential
from azure.identity import AzureCliCredential, ManagedIdentityCredential
from azureml.acft.common_components import get_logger_app
from azureml.core import Run, Workspace
from azureml.core.run import _OfflineRun

from typing import Union


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


def get_online_endpoint_key(mlclient_ws: MLClient, endpoint_name: str) -> str:
    """Return online endpoint primary key."""
    try:
        keys = mlclient_ws.online_endpoints.get_keys(endpoint_name)
        return keys.primary_key
    except Exception as e:
        logger.error(f"Exception in fetching online endpoint keys for endpoint name: {endpoint_name}. Error {e}")
        return None


def get_online_endpoint_url(mlclient_ws: MLClient, endpoint_name: str) -> str:
    """Return online endpoint URL for an endpoint name."""
    try:
        endpoint = mlclient_ws.online_endpoints.get(endpoint_name)
        return endpoint.scoring_uri
    except Exception as e:
        logger.error(
            f"Exception in fetching online endpoint scoring URL for endpoint name: {endpoint_name}. Error {e}")
        return None


def get_serverless_endpoint_key(mlclient_ws: MLClient, endpoint_name: str) -> str:
    """Return serverless endpoint primary key."""
    try:
        keys = mlclient_ws.serverless_endpoints.get_keys(endpoint_name)
        return keys.primary_key
    except Exception as e:
        logger.error(f"Exception in fetching serverless endpoint keys for endpoint name: {endpoint_name}. Error {e}")
        return None


def get_serverless_endpoint_url(mlclient_ws: MLClient, endpoint_name: str) -> str:
    """Return serverless endpoint URL for an endpoint name."""
    try:
        endpoint = mlclient_ws.serverless_endpoints.get(endpoint_name)
        return endpoint.scoring_uri
    except Exception as e:
        logger.error(
            f"Exception in fetching serverless endpoint scoring URL for endpoint name: {endpoint_name}. Error {e}")
        return None

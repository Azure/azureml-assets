# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Utility functions for the online evaluation context."""
import os
import re
import pandas as pd

from azure.ai.ml import MLClient
from azure.ai.ml.identity import AzureMLOnBehalfOfCredential
from azure.identity import ManagedIdentityCredential
from azure.monitor.query import LogsQueryClient
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def get_managed_identity_credentials():
    """Get the managed identity credentials."""
    client_id = os.environ.get('DEFAULT_IDENTITY_CLIENT_ID', None)
    credential = ManagedIdentityCredential(client_id=client_id)
    logger.info("ManagedIdentityCredential successfully loaded.")
    return credential


def get_user_identity_credentials():
    """Get the user identity or default credentials."""
    logger.info("Trying to load AzureMLOnBehalfOfCredential")
    credential = AzureMLOnBehalfOfCredential()
    logger.info("AzureMLOnBehalfOfCredential successfully loaded.")
    return credential


def get_credentials(use_managed_identity=True):
    """Get the credentials."""
    try:
        if use_managed_identity:
            logger.info("Initializing managed identity")
            credential = get_managed_identity_credentials()
        else:
            credential = get_user_identity_credentials()
        logger.info("Trying to fetch token for credentials")

    except Exception as e:
        logger.info("Error while loading credentials")
        raise e
    return credential


def get_app_insights_client(use_managed_identity):
    """Get the AppInsights client."""
    try:
        credential = get_credentials(use_managed_identity=use_managed_identity)
        async_logs_query_client = LogsQueryClient(credential)
    except Exception:
        safe_message = (
            "Not able to initialize AppInsights client. Please verify that the correct credentials have been provided")
        raise Exception(safe_message)
    return async_logs_query_client


# logger =
def extract_model_info(model_asset_id):
    """Extract model details from asset id."""
    # Define regular expressions for extracting information
    workspace_pattern = re.compile(
        r"azureml://locations/(?P<location>\w+)/workspaces/(?P<workspace>[\w-]+)/models/(?P<model_name>[\w-]+)/"
        r"versions/(?P<version>\d+)")
    registry_pattern = re.compile(
        r"azureml://registries/(?P<registry>[\w-]+)/models/(?P<model_name>[\w-]+)/versions/(?P<version>\d+)")

    # Try to match the input model asset ID with the patterns
    workspace_match = workspace_pattern.match(model_asset_id)
    registry_match = registry_pattern.match(model_asset_id)

    if workspace_match:
        # Extract information for workspace registered model
        info = workspace_match.groupdict()
        info['type'] = 'workspace_registered'
    elif registry_match:
        # Extract information for registry registered model
        info = registry_match.groupdict()
        info['type'] = 'registry_registered'
    else:
        # If neither pattern matches, return None
        return None
    return info


def get_mlclient(
        workspace_name: str = None, resource_group_name: str = None, subscription_id: str = None,
        registry_name: str = None
):
    """Return ML Client.

    :param workspace_name: Workspace name
    :type workspace_name: MLClient
    :param resource_group_name: resource group
    :type resource_group_name: str
    :param subscription_id: subscription ID
    :type subscription_id: str
    :param registry_name: registry name
    :type registry_name: str
    :return: MLClient object for workspace or registry
    :rtype: MLClient
    """
    credential = get_credentials(use_managed_identity=True)
    if registry_name is None:
        logger.info(f"Creating MLClient with sub: {subscription_id}, rg: {resource_group_name}, ws: {workspace_name}")
        return MLClient(
            credential=credential,
            subscription_id=subscription_id,
            resource_group_name=resource_group_name,
            workspace_name=workspace_name,
        )

    logger.info(f"Creating MLClient with registry name {registry_name}")
    return MLClient(credential=credential, registry_name=registry_name)


def is_input_data_empty(data_file_path):
    """Check if the input data is empty."""
    if not data_file_path:
        logger.info("Data file path is empty. Exiting.")
        return True

    df = pd.read_json(data_file_path, lines=True)
    if len(df) == 0:
        logger.info("Empty data in preprocessed file. Exiting.")
        return True
    return False

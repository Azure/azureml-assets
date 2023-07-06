# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Common utils."""

import os
from azure.ai.ml import MLClient
from azureml._common._error_definition import AzureMLError
from azureml._common.exceptions import AzureMLException
from azure.ai.ml.identity import AzureMLOnBehalfOfCredential
from azure.identity import ManagedIdentityCredential

from utils.logging_utils import get_logger
from utils.exceptions import NonMsiAttachedComputeError, UserIdentityMissingError


logger = get_logger(__name__)


def get_mlclient(registry_name: str = None):
    """Return ML Client."""
    has_obo_succeeded = False
    try:
        credential = AzureMLOnBehalfOfCredential()
        # Check if given credential can get token successfully.
        credential.get_token("https://management.azure.com/.default")
        has_obo_succeeded = True
    except Exception as ex:
        # Fall back to ManagedIdentityCredential in case AzureMLOnBehalfOfCredential does not work
        logger.exception(
            AzureMLException._with_error(
                AzureMLError.create(UserIdentityMissingError, exception=ex)
            )
        )

    if not has_obo_succeeded:
        try:
            msi_client_id = os.environ.get("DEFAULT_IDENTITY_CLIENT_ID")
            credential = ManagedIdentityCredential(client_id=msi_client_id)
            credential.get_token("https://management.azure.com/.default")
        except Exception as ex:
            raise AzureMLException._with_error(
                AzureMLError.create(NonMsiAttachedComputeError, exception=ex)
            )

    if registry_name is None:
        run = Run.get_context(allow_offline=False)
        ws = run.experiment.workspace
        return MLClient(
            credential=credential,
            subscription_id=ws._subscription_id,
            resource_group_name=ws._resource_group,
            workspace_name=ws._workspace_name,
        )
    logger.info(f"Creating MLClient with registry name {registry_name}")
    return MLClient(credential=credential, registry_name=registry_name)

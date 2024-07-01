# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Common utils."""

import os
import sys
from azure.ai.ml import MLClient
from azure.ai.ml.identity import AzureMLOnBehalfOfCredential
from azure.identity import ManagedIdentityCredential
from azureml._common._error_definition import AzureMLError
from azureml._common.exceptions import AzureMLException
from azureml.core.run import Run
from pathlib import Path
from subprocess import PIPE, run, STDOUT
from typing import Tuple
import re

from utils.logging_utils import get_logger
from utils.exceptions import UserIdentityMissingError, InvalidModelIDError


logger = get_logger(__name__)


def run_command(cmd: str, cwd: Path = "./") -> Tuple[int, str]:
    """Run the command and returns the result."""
    logger.info(cmd)
    result = run(
        cmd,
        cwd=cwd,
        shell=True,
        stdout=PIPE,
        stderr=STDOUT,
        encoding=sys.stdout.encoding,
        errors="ignore",
    )
    return result.returncode, result.stdout


def get_mlclient(registry_name: str = None):
    """Return ML Client."""
    has_msi_succeeded = False
    try:
        msi_client_id = os.environ.get("DEFAULT_IDENTITY_CLIENT_ID")
        credential = ManagedIdentityCredential(client_id=msi_client_id)
        credential.get_token("https://management.azure.com/.default")
        has_msi_succeeded = True
    except Exception:
        # Fall back to AzureMLOnBehalfOfCredential in case ManagedIdentityCredential does not work
        has_msi_succeeded = False
        logger.warning("ManagedIdentityCredential was not found in the compute. "
                       "Falling back to AzureMLOnBehalfOfCredential")

    if not has_msi_succeeded:
        try:
            credential = AzureMLOnBehalfOfCredential()
            # Check if given credential can get token successfully.
            credential.get_token("https://management.azure.com/.default")
        except Exception as ex:
            raise AzureMLException._with_error(AzureMLError.create(UserIdentityMissingError, exception=ex))

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


def get_model_name(model_id: str):
    """Return model name from model_id."""
    pattern = r"/(models)/([^/:]+)(:|/versions/)(\d+)$|:([^/:]+):(\d+)$"
    match = re.search(pattern, model_id)
    if match:
        return match.group(2) or match.group(5)
    else:
        raise AzureMLException._with_error(AzureMLError.create(InvalidModelIDError, model_id=model_id))

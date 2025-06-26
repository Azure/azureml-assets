# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Common utils."""

import os
import re
import sys
from azure.ai.ml import MLClient
from azure.ai.ml.identity import AzureMLOnBehalfOfCredential
from azure.identity import ManagedIdentityCredential
from azure.ai.ml.exceptions import ErrorTarget, ErrorCategory, MlException, ValidationException
from pathlib import Path
from subprocess import PIPE, run, STDOUT
from typing import Tuple

from utils.logging_utils import get_logger
from utils.exceptions import ModelImportErrorStrings


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
            message = ModelImportErrorStrings.USER_IDENTITY_MISSING_ERROR
            raise MlException(
                message=message.format(ex=ex), no_personal_data_message=message,
                error_category=ErrorCategory.SYSTEM_ERROR, target=ErrorTarget.IDENTITY,
                error=ex
            )
    try:
        subscription_id = os.environ['AZUREML_ARM_SUBSCRIPTION']
        resource_group = os.environ["AZUREML_ARM_RESOURCEGROUP"]
        workspace = os.environ["AZUREML_ARM_WORKSPACE_NAME"]
    except Exception as ex:
        message = "Failed to get AzureML ARM env variable : {ex}"
        raise MlException(
            message=message.format(ex=ex), no_personal_data_message=message,
            error_category=ErrorCategory.SYSTEM_ERROR, target=ErrorTarget.COMPONENT,
            error=ex
        )

    if registry_name is None:
        return MLClient(
            credential=credential,
            subscription_id=subscription_id,
            resource_group_name=resource_group,
            workspace_name=workspace,
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
        message = ModelImportErrorStrings.INVALID_MODEL_ID_ERROR
        raise ValidationException(
            message=message.format(model_id=model_id), no_personal_data_message=message,
            error_category=ErrorCategory.USER_ERROR, target=ErrorTarget.COMPONENT
        )


def get_model_name_version(model_id: str):
    """Return model name from model_id."""
    ws_pattern = r"azureml:(.+):(.+)"
    reg_pattern = r"azureml:\/\/.*\/models\/(.+)\/versions\/(.+)"

    # try registry pattern followed by ws pattern
    match = re.search(reg_pattern, model_id)
    if match:
        logger.info(f"registry asset URI, returning {match.group(1)}, {match.group(2)}")
        return match.group(1), match.group(2)

    match = re.search(ws_pattern, model_id)
    if match:
        logger.info(f"ws asset URI, returning {match.group(1)}, {match.group(2)}")
        return match.group(2) or match.group(5)

    message = "Unsupported model asset uri: {model_id}"
    logger.info(message.format(model_id=model_id))
    raise MlException(
        message=message.format(model_id=model_id), no_personal_data_message=message,
        error_category=ErrorCategory.USER_ERROR, target=ErrorTarget.COMPONENT
    )


def get_job_uri_from_input_run_assetId(assetID: str):
    """Return job asset ID post parsing run input asset ID.

    Expected Input assetID pattern:
        For model:
            azureml://locations/<loc>/workspaces/<ws-d>/model/azureml_<job-id>_output_<output-name>/versions/<v>
        For data:
            azureml://locations/<loc>/workspaces/<ws-id>/data/azureml_<job-id>_output_data_<output-name>/versions/<v>
    Corresponding pattern for job assetID:
        azureml://jobs/<parent_job_id>/outputs/mlflow_model_folder
    """
    input_pattern_str = r"azureml://locations/.+/workspaces/.+/(.+)/azureml_(.+)_output_(.+)/versions/(.+)"
    pattern = re.compile(input_pattern_str)
    match = pattern.search(assetID)
    if not match:
        logger.warning(f"Returning None, as {assetID} does not match with input asset ID pattern {input_pattern_str}")
        return None

    input_asset_type = match.group(1)
    input_parent_job_name = match.group(2)
    input_asset_name = match.group(3)
    if input_asset_type.lower() == "data":
        logger.info("Removing data_ prifix from asset name")
        input_asset_name = input_asset_name.replace("data_", "")
    return f"azureml://jobs/{input_parent_job_name}/outputs/{input_asset_name}"


def get_run_input_asset_id(input_name: str) -> str:
    """Get asset ID of the input provided to the pipeline component.

    Looks for environment variable 'AZUREML_JOB_INPUT_<INPUT_NAME>'
    """
    env_var = f"AZURE_ML_INPUT_{input_name}"
    asset_id = os.environ.get(env_var, None)

    if not asset_id:
        logger.warning(f"No asset ID found for input '{input_name}'. Expected env var: {env_var}")

    return asset_id


def get_job_asset_uri(input_name):
    """Get Job asset URI for input name."""
    input_asset_id = get_run_input_asset_id(input_name)
    return get_job_uri_from_input_run_assetId(input_asset_id)

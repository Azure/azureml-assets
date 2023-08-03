# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Validate Import Pipeline Parameters."""

import argparse
import os
from azure.ai.ml import MLClient
from azureml.model.mgmt.utils.exceptions import (
    ModelAlreadyExists,
    swallow_all_exceptions
)
from azureml._common.exceptions import AzureMLException
from azureml._common._error_definition.azureml_error import AzureMLError
from azureml.model.mgmt.config import AppName
from azureml.model.mgmt.utils.logging_utils import custom_dimensions, get_logger
from azure.ai.ml.identity import AzureMLOnBehalfOfCredential
from azure.identity import ManagedIdentityCredential

logger = get_logger(__name__)
custom_dimensions.app_name = AppName.VALIDATION_TRIGGER_IMPORT


def _get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    return parser


def get_mlclient(registry_name: str = None):
    """Return ML Client."""
    try:
        credential = AzureMLOnBehalfOfCredential()
        # Check if given credential can get token successfully.
        credential.get_token("https://management.azure.com/.default")

    except Exception:
        # Fall back to ManagedIdentityCredential in case AzureMLOnBehalfOfCredential does not work
        logger.warning("AzureMLOnBehalfOfCredential failed.")
        try:
            msi_client_id = os.environ.get("DEFAULT_IDENTITY_CLIENT_ID")
            credential = ManagedIdentityCredential(client_id=msi_client_id)
            credential.get_token("https://management.azure.com/.default")
        except Exception:
            logger.warning("ManagedIdentityCredential failed.")

    logger.info(f"Creating MLClient with registry name {registry_name}")
    return MLClient(credential=credential, registry_name=registry_name)


def validate_if_model_exists(model_id):
    """Validate if model exists in any of the registries."""
    registries_list = ["azureml-preview", "azureml", "azureml-meta"]

    for registry in registries_list:
        ml_client_registry = get_mlclient(registry_name=registry)
        REG_MODEL_ID = model_id.replace("/", "-")  # model name in registry doesn't contain '/'
        models = ml_client_registry.models.list(name=REG_MODEL_ID)
        print(f"models: {models}")
        if models:
            raise AzureMLException._with_error(
                AzureMLError.create(ModelAlreadyExists, model_id=model_id, registry=registry)
            )
        else:
            logger.info(f"Model {model_id} has not been imported into the registry. "
                        "Please continue importing the model.")


@swallow_all_exceptions(logger)
def validate():
    """Validate Import pipeline parameters."""
    parser = _get_parser()
    args, unknown_args_ = parser.parse_known_args()
    model_id = args.model_id
    validate_if_model_exists(model_id)


if __name__ == "__main__":
    validate()

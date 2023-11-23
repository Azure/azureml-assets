# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Validate Import Pipeline Parameters."""
import json, re, os
from azureml.model.mgmt.utils.logging_utils import custom_dimensions, get_logger
from azureml.model.mgmt.utils.exceptions import swallow_all_exceptions
from azureml.model.mgmt.config import AppName, LlamaHFModels, LlamaModels, llama_dict, ComponentVariables
from azureml.model.mgmt.utils.common_utils import get_mlclient
from azureml.model.mgmt.downloader import ModelSource
from mldesigner import Output, command_component
from azure.ai.ml import Input


VALID_MODEL_NAME_PATTERN = r"^[a-zA-Z0-9-]+$"
NEGATIVE_MODEL_NAME_PATTERN = r"[^a-zA-Z0-9-]"
logger = get_logger(__name__)
custom_dimensions.app_name = AppName.VALIDATION_TRIGGER_IMPORT


def validate_if_model_exists(model_id):
    """Validate if model exists in any of the registries."""
    registries_list = ["azureml", "azureml-meta"]

    # Hardcoding llama-hf model for now, to use llama models

    if LlamaHFModels.has_value(model_id):
        logger.warning(f"Lllama Model {model_id} with safe tensors is already present in registry. "
                       "Please use the same.")
        model_id = llama_dict[model_id]

    # Hardcoding check for llama models as names in registry do not contain meta-llama

    if LlamaModels.has_value(model_id):
        model_id = llama_dict[model_id]
        logger.info(f"Updated model_name = {model_id}")

    for registry in registries_list:
        try:
            ml_client_registry = get_mlclient(registry_name=registry)
        except Exception:
            logger.warning(f"Could not connect to registry {registry}")
            continue

        if not re.match(VALID_MODEL_NAME_PATTERN, model_id):
            # update model name to one supported for registration
            logger.info(f"Updating model name to match pattern `{VALID_MODEL_NAME_PATTERN}`")
            model_id = re.sub(NEGATIVE_MODEL_NAME_PATTERN, "-", model_id)
            logger.info(f"Updated model_name = {model_id}")

        try:
            model = ml_client_registry.models.get(name=model_id, label="latest")
        except Exception as e:
            logger.warning(f"Model with name - {model_id} is not available. Error: {e}")
            continue

        logger.info(f"model: {model}")
        if model:
            version = model.version
            url = f"https://ml.azure.com/registries/{registry}/models/{model_id}/version/{version}"
            model_info = {
                "id": model.id,
                "name": model.name,
                "version": model.version,
                "path": model.path,
                "flavors": model.flavors,
                "type": model.type,
                "properties": model.properties,
                "tags": model.tags,
                "description": model.description,
            }
            logger.warning(f"Model with name - {model_id} is already present in registry {registry} at {url} "
                           f"with details {model_info}. Please use the same.")
            
            return (True, model_info)
        else:
            logger.info(f"Model {model_id} has not been imported into the registry. "
                        "Please continue importing the model.")
    return (False, None)


@command_component
@swallow_all_exceptions(logger)
def validate(
    model_source: Input(type="string", required=True),
    model_id: Input(type="string", required=True),
    update_existing_model: Input(type="string", required=False),
    registration_details_folder: Output(type="uri_folder")
) -> Output(type="boolean", is_control=True):
    """Validate model import parameters."""

    if not ModelSource.has_value(model_source):
        raise Exception(f"Unsupported model source {model_source}")
    
    if update_existing_model.lower() == "false":
        if_model_exists, model_download_details = validate_if_model_exists(model_id)
        if if_model_exists:
            registrtaion_file  = os.path.join(registration_details_folder, ComponentVariables.REGISTRATION_DETAILS_JSON_FILE)
            with open(registrtaion_file, "w") as f:
                json.dump(model_download_details, f)
            return False
    return True

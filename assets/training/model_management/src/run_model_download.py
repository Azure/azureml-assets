# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Run Model downloader module."""

import json
import re
from azureml.model.mgmt.config import AppName, LlamaHFModels, LlamaModels, llama_dict
from azureml.model.mgmt.downloader import download_model, ModelSource
from azureml.model.mgmt.utils.exceptions import swallow_all_exceptions
from azureml.model.mgmt.utils.logging_utils import custom_dimensions, get_logger
from azureml.model.mgmt.utils.common_utils import get_mlclient
from mldesigner import Output, command_component
from azure.ai.ml import Input

VALID_MODEL_NAME_PATTERN = r"^[a-zA-Z0-9-]+$"
NEGATIVE_MODEL_NAME_PATTERN = r"[^a-zA-Z0-9-]"
logger = get_logger(__name__)
custom_dimensions.app_name = AppName.DOWNLOAD_MODEL


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
            logger.warning(f"Model with name - {model_id} is already present in registry {registry} at {url}")
            return True
        else:
            logger.info(f"Model {model_id} has not been imported into the registry. "
                        "Please continue importing the model.")
    return False


@command_component()
@swallow_all_exceptions(logger)
def run_download_model(
    model_source: Input(type="str", required=True),
    model_id: Input(type="str", required=True),
    validation_info: Input(type="str", required=False),  # Dummy input to create dependency on validation
    update_existing_model: Input(type="str", required=False, default="False"),
    model_download_metadata: Output(type="str"),
    model_output_dir: Output(type="str"),
) -> Output(type="bool", is_control=True):
    """Run model download."""
    update_existing_model = update_existing_model.lower()

    if not ModelSource.has_value(model_source):
        raise Exception(f"Unsupported model source {model_source}")

    logger.info(f"Model source: {model_source}")
    logger.info(f"Model id: {model_id}")
    logger.info(f"Update existing model: {update_existing_model}")

    if update_existing_model == "false":
        if validate_if_model_exists(model_id):
            return False

    logger.info("Downloading model")
    model_download_details = download_model(
        model_source=model_source, model_id=model_id, download_dir=model_output_dir
    )

    with open(model_download_metadata, "w") as f:
        json.dump(model_download_details, f)

    logger.info("Download completed.")

    return True

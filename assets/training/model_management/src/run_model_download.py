# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Run Model downloader module."""

import argparse
import json
import re
from azureml.model.mgmt.config import AppName, LlamaHFModels, LlamaModels, llama_dict
from azureml.model.mgmt.downloader import download_model, ModelSource
from azureml.model.mgmt.utils.exceptions import swallow_all_exceptions, ModelImportErrorStrings
from azureml.model.mgmt.utils.logging_utils import custom_dimensions, get_logger
from azureml.model.mgmt.utils.common_utils import get_mlclient
from azure.ai.ml.exceptions import ErrorTarget, ErrorCategory, MlException

VALID_MODEL_NAME_PATTERN = r"^[a-zA-Z0-9-]+$"
NEGATIVE_MODEL_NAME_PATTERN = r"[^a-zA-Z0-9-]"
logger = get_logger(__name__)
custom_dimensions.app_name = AppName.DOWNLOAD_MODEL


def _get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-source", required=True, help="Model source ")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-download-metadata", required=True, help="Model source info file path")
    parser.add_argument("--model-output-dir", required=True, help="Model download directory")
    parser.add_argument("--update-existing-model", required=False, default='false', help="Update existing model")
    parser.add_argument("--validation-info", required=False, help="Validation info")
    parser.add_argument("--token", required=False, help="Token to access the private models or authenticate the user.")
    return parser


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
            message = ModelImportErrorStrings.MODEL_ALREADY_EXISTS.format(
                model_id=model_id, registry=registry, url=url
            )
            raise MlException(
                message=message, no_personal_data_message=message,
                error_category=ErrorCategory.USER_ERROR, target=ErrorTarget.COMPONENT
            )
        else:
            logger.info(f"Model {model_id} has not been imported into the registry. "
                        "Please continue importing the model.")


@swallow_all_exceptions(logger)
def run():
    """Run model download."""
    parser = _get_parser()
    args, unknown_args_ = parser.parse_known_args()

    model_source = args.model_source
    model_id = args.model_id
    model_download_metadata_path = args.model_download_metadata
    model_output_dir = args.model_output_dir
    update_existing_model = args.update_existing_model.lower()
    token = args.token

    if not ModelSource.has_value(model_source):
        raise Exception(f"Unsupported model source {model_source}")

    logger.info(f"Model source: {model_source}")
    logger.info(f"Model id: {model_id}")
    logger.info(f"Update existing model: {update_existing_model}")

    if update_existing_model == "false":
        validate_if_model_exists(model_id)

    logger.info("Downloading model")
    model_download_details = download_model(
        model_source=model_source, model_id=model_id, download_dir=model_output_dir, token=token
    )

    with open(model_download_metadata_path, "w") as f:
        json.dump(model_download_details, f)

    logger.info("Download completed.")


if __name__ == "__main__":
    run()

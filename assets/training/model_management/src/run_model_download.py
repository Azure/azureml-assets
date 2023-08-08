# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Run Model downloader module."""

import argparse
import json
import re
from azureml.model.mgmt.config import AppName
from azureml.model.mgmt.downloader import download_model, ModelSource
from azureml.model.mgmt.utils.exceptions import swallow_all_exceptions, ModelAlreadyExists
from azureml.model.mgmt.utils.logging_utils import custom_dimensions, get_logger
from azureml.model.mgmt.utils.common_utils import get_mlclient
from azureml._common.exceptions import AzureMLException
from azureml._common._error_definition.azureml_error import AzureMLError

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
    parser.add_argument("--update-existing-model", required=False, help="Update existing model")
    return parser


def validate_if_model_exists(model_id):
    """Validate if model exists in any of the registries."""
    registries_list = ["azureml", "azureml-meta"]

    for registry in registries_list:
        try:
            ml_client_registry = get_mlclient(registry_name=registry)
        except Exception:
            continue

        if not re.match(VALID_MODEL_NAME_PATTERN, model_id):
        # update model name to one supported for registration
            logger.info(f"Updating model name to match pattern `{VALID_MODEL_NAME_PATTERN}`")
            model_id = re.sub(NEGATIVE_MODEL_NAME_PATTERN, "-", model_id)
            logger.info(f"Updated model_name = {model_id}")

        models = ml_client_registry.models.list(name=model_id)
        print(f"models: {models}")
        if models:
            version = models[0].version
            url = f"https://ml.azure.com/registries/{registry}/models/{model_id}/version/{version}"
            raise AzureMLException._with_error(
                AzureMLError.create(ModelAlreadyExists, model_id=model_id, registry=registry, url=url)
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

    if not ModelSource.has_value(model_source):
        raise Exception(f"Unsupported model source {model_source}")

    logger.info(f"Model source: {model_source}")
    logger.info(f"Model id: {model_id}")
    logger.info(f"Update existing model: {update_existing_model}")

    if update_existing_model == "false":
        validate_if_model_exists(model_id)

    logger.info("Downloading model")
    model_download_details = download_model(
        model_source=model_source, model_id=model_id, download_dir=model_output_dir
    )

    with open(model_download_metadata_path, "w") as f:
        json.dump(model_download_details, f)

    logger.info("Download completed.")


if __name__ == "__main__":
    run()

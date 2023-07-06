# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Run Model Registration module."""

import argparse
import json
import os
import shutil
import yaml

from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Model
from azureml._common._error_definition import AzureMLError
from azureml._common.exceptions import AzureMLException

from utils.common_utils import get_mlclient
from utils.config import AppName
from utils.logging_utils import custom_dimensions, get_logger
from utils.exceptions import (
    swallow_all_exceptions,
    UnSupportedModelTypeError,
    MissingModelNameError,
)


SUPPORTED_MODEL_ASSET_TYPES = [AssetTypes.CUSTOM_MODEL, AssetTypes.MLFLOW_MODEL]
MLFLOW_MODEL_FOLDER = "mlflow_model_folder"

logger = get_logger(__name__)
custom_dimensions.app_name = AppName.REGISTER_MODEL


def parse_args():
    """Return arguments."""
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--model_path", type=str, help="Directory containing model files")
    parser.add_argument(
        "--model_type",
        type=str,
        default="mlflow_model",
        help="Type of model you want to register",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="Name to use for the registered model. If it already exists, the version will be auto incremented.",
    )
    parser.add_argument(
        "--model_description",
        type=str,
        help="Description of the model that will be shown in registry/workspace",
        default=None,
    )
    parser.add_argument(
        "--registry_name",
        type=str,
        help="Name of the asset registry where the model will be registered",
        default=None,
    )
    parser.add_argument(
        "--registration_details",
        type=str,
        help="JSON file into which model registration details will be written",
    )
    parser.add_argument(
        "--model_download_metadata",
        type=str,
        help="JSON file containing metadata related to the downloaded model",
        default=None,
    )
    parser.add_argument(
        "--model_metadata",
        type=str,
        help="JSON/YAML file that contains model metadata confirming to Model V2",
        default=None,
    )
    parser.add_argument(
        "--model_version",
        type=str,
        help="Model version in workspace/registry. If model with same version exists,version will be auto incremented",
        default=None,
    )
    parser.add_argument(
        "--model_import_job_path",
        type=str,
        help="JSON file that contains the job path of model to have lineage.",
        default=None,
    )
    args = parser.parse_args()
    logger.info(f"Args received {args}")
    return args


def is_model_available(ml_client, model_name, model_version):
    """Return true if model is available else false."""
    is_available = True
    try:
        ml_client.models.get(name=model_name, version=model_version)
    except Exception as e:
        logger.exception(f"Model with name - {model_name} and version - {model_version} is not available. Error: {e}")
        is_available = False
    return is_available


@swallow_all_exceptions(logger)
def main():
    """Run main function."""
    args = parse_args()
    model_name = args.model_name
    model_type = args.model_type
    model_description = args.model_description
    registry_name = args.registry_name
    model_path = args.model_path
    registration_details = args.registration_details
    model_version = args.model_version
    tags, properties, flavors = {}, {}, {}

    ml_client = get_mlclient(registry_name)

    model_download_metadata = {}
    if args.model_download_metadata:
        with open(args.model_download_metadata) as f:
            model_download_metadata = json.load(f)
            model_name = model_name or model_download_metadata.get("name", "").replace("/", "-")
            tags = model_download_metadata.get("tags", tags)
            properties = model_download_metadata.get("properties", properties)

    # Updating tags and properties with value provided in metadata file
    if args.model_metadata:
        with open(args.model_metadata, "r") as stream:
            metadata = yaml.safe_load(stream)
            tags.update(metadata.get("tags", {}))
            properties.update(metadata.get("properties", {}))
            model_description = metadata.get("description", model_description)
            model_type = metadata.get("type", model_type)
            flavors = metadata.get("flavors", flavors)

    # validations
    if model_type not in SUPPORTED_MODEL_ASSET_TYPES:
        raise AzureMLException._with_error(
                AzureMLError.create(UnSupportedModelTypeError, model_type=model_type)
            )

    if not model_name:
        raise AzureMLException._with_error(
            AzureMLError.create(MissingModelNameError)
        )

    # check if we can have lineage and update the model path for ws import
    if not registry_name and args.model_import_job_path:
        logger.info("Using model output of previous job as run lineage to register the model")
        with open(args.model_import_job_path) as f:
            model_import_job_path = json.load(f)
        model_path = model_import_job_path.get("path", model_path)
    elif model_type == AssetTypes.MLFLOW_MODEL:
        if not os.path.exists(os.path.join(model_path, MLFLOW_MODEL_FOLDER)):
            logger.info(f"Making sure, model parent directory is `{MLFLOW_MODEL_FOLDER}`")
            shutil.copytree(model_path, MLFLOW_MODEL_FOLDER, dirs_exist_ok=True)
            model_path = MLFLOW_MODEL_FOLDER
        mlmodel_path = os.path.join(model_path, "MLmodel")
        logger.info(f"MLModel path: {mlmodel_path}")
        with open(mlmodel_path, "r") as stream:
            metadata = yaml.safe_load(stream)
            flavors = metadata.get('flavors', flavors)

    if not model_version or is_model_available(ml_client, model_name, model_version):
        # hack to get current model versions in registry
        model_version = "1"
        models_list = []
        try:
            models_list = ml_client.models.list(name=model_name)
            if models_list:
                max_version = (max(models_list, key=lambda x: int(x.version))).version
                model_version = str(int(max_version) + 1)
        except Exception:
            exception_msg = (
                f"Error in listing versions for model {model_name}."
                "Trying to register model with version '1'"
            )
            logger.exception(exception_msg)

    model = Model(
        name=model_name,
        version=model_version,
        type=model_type,
        path=model_path,
        tags=tags,
        properties=properties,
        flavors=flavors,
        description=model_description,
    )

    # register the model in workspace or registry
    logger.info(f"Registering model {model_name} with version {model_version}.")
    registered_model = ml_client.models.create_or_update(model)
    logger.info(f"Model registered. AssetID : {registered_model.id}")
    # Registered model information
    model_info = {
        "id": registered_model.id,
        "name": registered_model.name,
        "version": registered_model.version,
        "path": registered_model.path,
        "flavors": registered_model.flavors,
        "type": registered_model.type,
        "properties": registered_model.properties,
        "tags": registered_model.tags,
        "description": registered_model.description,
    }
    json_object = json.dumps(model_info, indent=4)

    with open(registration_details, "w") as outfile:
        outfile.write(json_object)
    logger.info("Saved model registration details in output json file.")


# run script
if __name__ == "__main__":
    main()

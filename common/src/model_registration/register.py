# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Run Model Registration module."""

import argparse
import json
import os
import shutil

from azure.ai.ml import MLClient
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Model
from azureml.core import Run
from azure.identity import ManagedIdentityCredential
from pathlib import Path


SUPPORTED_MODEL_ASSET_TYPES = [AssetTypes.CUSTOM_MODEL, AssetTypes.MLFLOW_MODEL]


def parse_args():
    """Return arguments."""
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument(
        "--model_path", type=str, help="Directory containing model files"
    )
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
        help="Text File into which model registration details will be written",
    )
    parser.add_argument(
        "--model_info_path",
        type=str,
        help="Json file containing metadata related to the model",
        default=None,
    )

    args = parser.parse_args()
    print("args received ", args)
    return args


def get_ml_client(registry_name):
    """Return ML Client."""
    msi_client_id = os.environ.get("DEFAULT_IDENTITY_CLIENT_ID")
    credential = ManagedIdentityCredential(client_id=msi_client_id)
    if registry_name is None:
        run = Run.get_context(allow_offline=False)
        ws = run.experiment.workspace
        return MLClient(
            credential=credential,
            subscription_id=ws._subscription_id,
            resource_group_name=ws._resource_group,
            workspace_name=ws._workspace_name,
        )
    return MLClient(credential=credential, registry_name=registry_name)


def main(args):
    """Run main function."""
    model_name = args.model_name
    model_type = args.model_type
    model_description = args.model_description
    registry_name = args.registry_name
    model_path = args.model_path
    model_info_path = args.model_info_path
    registration_details = args.registration_details

    ml_client = get_ml_client(registry_name)

    model_info = {}
    if model_info_path:
        with open(model_info_path) as f:
            model_info = json.load(f)

    model_name = model_info.get("name", model_name)
    model_type = model_info.get("type", model_type)
    model_description = model_info.get("description", model_description)
    tags = model_info.get("tags", {})
    properties = model_info.get("properties", {})

    # validations
    if not model_type in SUPPORTED_MODEL_ASSET_TYPES:
        raise Exception(f"Unsupported model type {model_type}")

    if not model_name:
        raise Exception(
            "Model name is a required parameter. Provide model_name in the component input or in the model_info JSON"
        )

    if model_type == "mlflow_model":
        # Make sure parent directory is mlflow_model_folder for mlflow model
        shutil.copytree(model_path, "mlflow_model_folder", dirs_exist_ok=True)
        model_path = "mlflow_model_folder"

    # hack to get current model versions in registry
    model_version = "1"
    models_list = []
    try:
        models_list = ml_client.models.list(name=model_name)
        if models_list:
            max_version = (max(models_list, key=lambda x:x.version)).version
            model_version = str(int(max_version) + 1)
    except Exception:
        print(f"Error in listing versions for model {model_name}. Trying to register model with version '1'.")

    model = Model(
        name=model_name,
        version=model_version,
        type=model_type,
        path=model_path,
        tags=tags,
        properties=properties,
    )

    # register the model in workspace or registry
    registered_model = ml_client.models.create_or_update(model)
    
    print(f"model => {registered_model}")
    # write registered model id which will be fetched by deployment component
    (Path(registration_details)).write_text(registered_model.id)


# run script
if __name__ == "__main__":
    main(parse_args())

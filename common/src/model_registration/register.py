# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Run Model Registration module."""

import argparse
import json
import os
import shutil
import yaml

from azure.ai.ml import MLClient
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Model
from azure.identity import ManagedIdentityCredential
from azure.ai.ml.identity import AzureMLOnBehalfOfCredential
from azureml.core import Run
from applicationinsights import TelemetryClient


SUPPORTED_MODEL_ASSET_TYPES = [AssetTypes.CUSTOM_MODEL, AssetTypes.MLFLOW_MODEL]
MLFLOW_MODEL_FOLDER = "mlflow_model_folder"

tc = None


def init_tc():
    """Initialize telemetry client."""
    global tc
    if tc is None:
        try:
            tc = TelemetryClient("71b954a8-6b7d-43f5-986c-3d3a6605d803")
        except Exception as e:
            print(f"Exception while initializing app insights: {e}")


def tc_log(message):
    """Log message to app insights."""
    global tc
    try:
        print(message)
        tc.track_event(name="FM_import_pipeline_debug_logs", properties={"message": message})
        tc.flush()
    except Exception as e:
        print(f"Exception while logging to app insights: {e}")


def tc_exception(e, message):
    """Log exception to app insights."""
    global tc
    try:
        tc.track_exception(value=e.__class__, properties={"exception": message})
        tc.flush()
    except Exception as e:
        print(f"Exception while logging exception to app insights: {e}")


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
    tc_log(f"Args received {args}")
    return args


def get_ml_client(registry_name):
    """Return ML Client."""
    has_obo_succeeded = False
    try:
        credential = AzureMLOnBehalfOfCredential()
        # Check if given credential can get token successfully.
        credential.get_token("https://management.azure.com/.default")
        has_obo_succeeded = True
    except Exception as ex:
        # Fall back to ManagedIdentityCredential in case AzureMLOnBehalfOfCredential does not work
        tc_exception(ex, f"Failed to get OBO credentials - {ex}")

    if not has_obo_succeeded:
        try:
            msi_client_id = os.environ.get("DEFAULT_IDENTITY_CLIENT_ID")
            credential = ManagedIdentityCredential(client_id=msi_client_id)
            credential.get_token("https://management.azure.com/.default")
        except Exception as ex:
            tc_exception(ex, f"Failed to get MSI credentials : {ex}")
            raise (f"Failed to get MSI credentials : {ex}")

    if registry_name is None:
        run = Run.get_context(allow_offline=False)
        ws = run.experiment.workspace
        return MLClient(
            credential=credential,
            subscription_id=ws._subscription_id,
            resource_group_name=ws._resource_group,
            workspace_name=ws._workspace_name,
        )
    tc_log(f"Creating MLClient with registry name {registry_name}")
    return MLClient(credential=credential, registry_name=registry_name)


def is_model_available(ml_client, model_name, model_version):
    """Return true if model is available else false."""
    is_available = True
    try:
        ml_client.models.get(name=model_name, version=model_version)
    except Exception as e:
        tc_exception(e, f"Model with name - {model_name} and version - "
                     "{model_version} is not available. Error: {e}")
        is_available = False
    return is_available


def main(args):
    """Run main function."""
    model_name = args.model_name
    model_type = args.model_type
    model_description = args.model_description
    registry_name = args.registry_name
    model_path = args.model_path
    registration_details = args.registration_details
    model_version = args.model_version
    tags, properties, flavors = {}, {}, {}

    init_tc()

    ml_client = get_ml_client(registry_name)

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
        tc_log(f"Unsupported model type {model_type}")
        raise Exception(f"Unsupported model type {model_type}")

    if not model_name:
        tc_log("Missing Model Name. Provide model_name as input or in the model_download_metadata JSON")
        raise Exception("Missing Model Name. Provide model_name as input or in the model_download_metadata JSON")

    # check if we can have lineage and update the model path for ws import
    if not registry_name and args.model_import_job_path:
        tc_log("Using model output of previous job as run lineage to register the model")
        with open(args.model_import_job_path) as f:
            model_import_job_path = json.load(f)
        model_path = model_import_job_path.get("path", model_path)
    elif model_type == AssetTypes.MLFLOW_MODEL:
        if not os.path.exists(os.path.join(model_path, MLFLOW_MODEL_FOLDER)):
            tc_log(f"Making sure, model parent directory is `{MLFLOW_MODEL_FOLDER}`")
            shutil.copytree(model_path, MLFLOW_MODEL_FOLDER, dirs_exist_ok=True)
            model_path = MLFLOW_MODEL_FOLDER
        mlmodel_path = os.path.join(model_path, "MLmodel")
        tc_log(f"MLModel path: {mlmodel_path}")
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
            tc_log(f"Error in listing versions for model {model_name}. Trying to register model with version '1'")

    tc_log("Arguments:")
    tc_log(f"Model type: {model_type}")

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
    tc_log(f"Registering model {model_name} with version {model_version}.")
    registered_model = ml_client.models.create_or_update(model)
    tc_log(f"Model registered. AssetID : {registered_model.id}")
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
    tc_log("Saved model registration details in output json file.")


# run script
if __name__ == "__main__":
    main(parse_args())

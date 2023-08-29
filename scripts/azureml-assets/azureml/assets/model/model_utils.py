# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Model utils Operations Class."""

import copy
import azureml.assets as assets
from azure.identity import AzureCliCredential
from azure.ai.ml import load_model, MLClient
from azure.ai.ml._utils._registry_utils import get_asset_body_for_registry_storage, get_registry_client
from azureml.assets.util import logger
from azureml.assets.config import PathType
from azureml.assets.model.download_utils import copy_azure_artifacts, download_git_model
from azureml.assets.deployment_config import AssetVersionUpdate


class RegistryUtils:
    """Registry utils."""

    RETRY_COUNT = 3

    def get_registry_data_reference(registry_name: str, model_name: str, model_version: str):
        """Fetch data reference for asset in the registry."""
        asset_id = f"azureml://registries/{registry_name}/models/{model_name}/versions/{model_version}"
        logger.print(f"getting data reference for asset {asset_id}")
        for cnt in range(1, RegistryUtils.RETRY_COUNT + 1):
            try:
                response = RegistryUtils._get_temp_data_ref(registry_name, model_name, model_version)
                blob_uri = response.blob_reference_for_consumption.blob_uri
                sas_uri = response.blob_reference_for_consumption.credential.additional_properties["sasUri"]
                if not blob_uri or not sas_uri:
                    raise Exception("Error in fetching BLOB or SAS URI")
                return blob_uri, sas_uri
            except Exception as e:
                logger.log_error(f"Exception in fetching data reference. Try #{cnt}. Error: {e}")
        else:
            raise Exception(f"Unable to fetch data reference for asset {asset_id}")

    def _get_temp_data_ref(registry_name, model_name, model_version):
        try:
            credential = AzureCliCredential()
            body = get_asset_body_for_registry_storage(registry_name, "models", model_name, model_version)
            registry_client, resource_group, _ = get_registry_client(credential, registry_name)
            response = registry_client.temporary_data_references.create_or_get_temporary_data_reference(
                name=model_name,
                version=model_version,
                resource_group_name=resource_group,
                registry_name=registry_name,
                body=body,
            )
            return response
        except Exception as e:
            logger.log_error(f"exception in fetching data reference: {e}")


class ModelAsset:
    """Asset class for model."""

    def __init__(self, spec_path, model_config, registry_name, temp_dir):
        """Initialize model asset."""
        self._spec_path = spec_path
        self._model_config = model_config
        self._model.description = model_config.description
        self._registry_name = registry_name
        self._temp_dir = temp_dir

        try:
            self._model = load_model(spec_path)
            self._model.type = model_config.type.value
        except Exception as e:
            logger.error(f"Error in loading model spec file at {spec_path}: {e}")
            return False

    def _publish_to_registry(self):
        src_uri = self._model_config.path.uri
        if self._model_config.path.type == PathType.GIT:
            # download model locally
            src_uri = self._temp_dir
            logger.print(f"Cloning model files from git {self._model_config.path.uri}")
            success = download_git_model(self._model_config.path.uri, src_uri)
            logger.print("Completed cloning")
            if not success:
                raise Exception(f"Cloning uri {self._model_config.path.uri} failed")

        try:
            # copy to registry blobstorage
            logger.print("get data ref for registry storage upload")
            blob_uri, sas_uri = RegistryUtils.get_registry_data_reference(
                self._registry_name, self._model.name, self._model.version
            )
            success = copy_azure_artifacts(src_uri, sas_uri)
            if not success:
                raise Exception("blobstorage copy failed.")
            logger.print("Successfully copied model artifacts to registry storage")
            return blob_uri
        except Exception as e:
            raise Exception(f"Error in copying artifacts to registry storage. Error {e}")


class MLFlowModelAsset(ModelAsset):
    """Asset class for MLflow model."""

    MLMODEL_FILE_NAME = "MLmodel"
    MLFLOW_MODEL_PATH = "mlflow_model_folder"

    def __init__(self, spec_path, model_config, registry_name, temp_dir):
        """Initialize Mlflow model asset."""
        super().__init__(spec_path, model_config, registry_name, temp_dir)

    def prepare_model(self):
        """Prepare model for publish."""
        model_registry_path = self._publish_to_registry()
        self._model.path = model_registry_path + "/" + MLFlowModelAsset.MLFLOW_MODEL_PATH
        return self._model


class CustomModelAsset(ModelAsset):
    """Asset class for custom model."""

    def __init__(self, spec_path, model_config, registry_name, temp_dir):
        """Initialize custom model asset."""
        super().__init__(spec_path, model_config, registry_name, temp_dir)

    def prepare_model(self):
        """Prepare model for publish."""
        model_registry_path = self._publish_to_registry()
        self._model.path = model_registry_path
        return self._model


def prepare_model(spec_path, model_config, registry_name, temp_dir):
    """Prepare model for publish."""
    try:
        logger.print(f"Model type: {model_config.type}")
        if model_config.type == assets.ModelType.CUSTOM:
            model_asset = CustomModelAsset(spec_path, model_config, registry_name, temp_dir)
        elif model_config.type == assets.ModelType.MLFLOW:
            model_asset = MLFlowModelAsset(spec_path, model_config, registry_name, temp_dir)
        else:
            logger.log_error(f"Model type {model_config.type.value} not supported")
            return False
        model = model_asset.prepare_model()
        return model, True
    except Exception as e:
        logger.log_error(f"prepare model failed for {spec_path}. Error {e}")
        return None, False


def update_model_metadata(
    mlclient: MLClient,
    model_name: str,
    model_version: str,
    update: AssetVersionUpdate,
):
    """Update the mutable metadata of already registered Model."""
    try:
        model = mlclient.models.get(name=model_name, version=model_version)

        need_update = False
        updated_tags = copy.deepcopy(model.tags)
        if update.tags:
            # Replace tags
            if update.tags.replace is not None:
                updated_tags = update.tags.replace
            elif update.tags.add is not None:
                for k, v in update.tags.add.items():
                    updated_tags[k] = v
            elif update.tags.delete is not None:
                for k in update.tags.delete:
                    updated_tags.pop(k, None)

        if updated_tags != model.tags:
            logger.print("tags has been updated.")
            model.tags = updated_tags
            need_update = True

        if model.description != update.description:
            logger.print("description has been updated")
            model.description = update.description
            need_update = True

        if not need_update:
            logger.print(f"No update found for model {model_name}. Skipping")
        else:
            mlclient.models.create_or_update(model)
            logger.print(f"Model metadata updated successfully for {model_name}")
    except Exception as e:
        logger.log_error(f"Failed to update metadata for model : {model_name} : {e}")

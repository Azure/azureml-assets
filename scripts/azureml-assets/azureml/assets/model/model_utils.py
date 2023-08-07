# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Model utils Operations Class."""

import os
import azureml.assets as assets
from azure.identity import DefaultAzureCredential
from azure.ai.ml import load_model
from azure.ai.ml._utils._registry_utils import get_asset_body_for_registry_storage, get_registry_client
from azureml.assets.util import logger
from azureml.assets.config import PathType
from azureml.assets.model.download_utils import copy_azure_artifacts, download_git_model
from pathlib import Path


class RegistryUtils:
    """Registry utils."""

    RETRY_COUNT = 3

    def get_registry_data_reference(registry_name: str, model_name: str, model_version: str):
        """Fetch data reference for asset in the registry."""
        cnt, blob_uri, sas_uri = 0, None, None
        asset_id = f"azureml://registries/{registry_name}/models/{model_name}/versions/{model_version}"
        logger.print(f"getting data reference for asset {asset_id}")
        while cnt < RegistryUtils.RETRY_COUNT:
            try:
                response = RegistryUtils._get_temp_data_ref(registry_name, model_name, model_version)
                blob_uri = response.blob_reference_for_consumption.blob_uri
                sas_uri = response.blob_reference_for_consumption.credential.additional_properties["sasUri"]
                if not blob_uri or not sas_uri:
                    raise Exception("Error in fetching BLOB or SAS URI")
                break
            except Exception as e:
                cnt += 1
                logger.log_error(f"Exception in fetching data reference. Retry count {cnt}. Error: {e}")
        if not blob_uri or not sas_uri:
            raise Exception(f"Unable to fetch data reference for asset {asset_id}")
        return blob_uri, sas_uri

    def _get_temp_data_ref(registry_name, model_name, model_version):
        try:
            credential = DefaultAzureCredential()
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
        self._registry_name = registry_name
        self._temp_dir = temp_dir

        try:
            self._model = load_model(spec_path)
            self._model.type = model_config.type.value
        except Exception as e:
            logger.error(f"Error in loading model spec file at {spec_path}: {e}")
            return False

        model_description_file_path = Path(spec_path).parent / model_config.description
        logger.print(f"model_description_file_path {model_description_file_path}")
        if os.path.exists(model_description_file_path):
            with open(model_description_file_path) as f:
                model_description = f.read()
                self._model.description = model_description
        else:
            logger.print("description file does not exist")

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

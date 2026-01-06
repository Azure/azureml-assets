# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Model utils Operations Class."""

import copy
import azureml.assets as assets
from typing import Tuple, Union
from azure.ai.ml import load_model, load_data, MLClient, operations as ops
from azure.ai.ml._utils._registry_utils import get_asset_body_for_registry_storage
from azureml.assets.util import logger
from azureml.assets.config import AssetType, Config, DataConfig, ModelConfig
from azureml.assets.util.util import resolve_from_file_for_asset
from azureml.assets.config import PathType
from azureml.assets.model.download_utils import CopyUpdater, copy_azure_artifacts, download_git_model
from azureml.assets.deployment_config import AssetVersionUpdate
from pathlib import Path


class RegistryUtils:
    """Registry utils."""

    RETRY_COUNT = 3

    def pluralize_asset_type(asset_type: Union[AssetType, str]) -> str:
        """Return pluralized asset type."""
        # Convert to string if enum
        if isinstance(asset_type, AssetType):
            asset_type = asset_type.value
        return f"{asset_type}s" if asset_type != "data" else asset_type

    def get_operations_from_type(asset_type: AssetType, ml_client: MLClient) -> Union[
                                 ops.ComponentOperations, ops.DataOperations, ops.EnvironmentOperations,
                                 ops.ModelOperations]:
        """Get MLCLient operations related to an asset type.

        Args:
            asset_type (AssetType): Asset type.
            ml_client (MLClient): ML client.

        Returns:
            Union[ops.ComponentOperations, ops.DataOperations, ops.EnvironmentOperations,
                ops.ModelOperations]: Operations object.
        """
        if asset_type == AssetType.COMPONENT:
            return ml_client.components
        elif asset_type == AssetType.DATA:
            return ml_client.data
        elif asset_type == AssetType.ENVIRONMENT:
            return ml_client.environments
        elif asset_type == AssetType.MODEL:
            return ml_client.models

    def publish_to_registry(ml_client: MLClient, extra_config: Config, asset_name: str, asset_version: str,
                            asset_type: assets.AssetType, temp_dir: Path, copy_updater: CopyUpdater = None,
                            output_level: str = "essential"):
        """Copy artifacts to registry storage.

        Args:
            ml_client (MLClient): ML client.
            extra_config (Config): ModelConfig or DataConfig object containing external storage info.
            asset_name (str): Asset name.
            asset_version (str): Asset version.
            asset_type (assets.AssetType): Asset type.
            temp_dir (Path): temp dir for asset operation.
            copy_updater (CopyUpdater): CopyUpdater object to update files during azcopy.
            output_level (str, optional): Output verbosity level parameter for azcopy. Defaults to "essential".
        """
        src_uri = extra_config.path.uri
        if extra_config.path.type == PathType.GIT:
            # download model locally (this is supported for models)
            src_uri = temp_dir
            logger.print(f"Cloning {asset_type} files from git {extra_config.path.uri}")
            success = download_git_model(extra_config.path.uri, src_uri)
            logger.print("Completed cloning")
            if not success:
                raise Exception(f"Cloning uri {extra_config.path.uri} failed")

        try:
            # copy to registry blobstorage (this is supported for models & data assets)
            logger.print("get data ref for registry storage upload")

            blob_uri, sas_uri = RegistryUtils.get_registry_data_reference(
                asset_name, asset_version, asset_type, ml_client
            )

            success = copy_azure_artifacts(src_uri=src_uri, dstn_uri=sas_uri, copy_updater=copy_updater,
                                           output_level=output_level)
            if not success:
                raise Exception("blobstorage copy failed.")
            logger.print(f"Successfully copied {asset_type.value} artifacts to registry storage")
            return blob_uri
        except Exception as e:
            raise Exception(f"Error in copying artifacts to registry storage. Error {e}")

    def get_registry_data_reference(asset_name: str, asset_version: str, asset_type: assets.AssetType,
                                    ml_client: MLClient) -> Tuple[str, str]:
        """Fetch data reference for asset in the registry."""
        asset_type_pluralized = RegistryUtils.pluralize_asset_type(asset_type)
        operations = RegistryUtils.get_operations_from_type(asset_type=asset_type, ml_client=ml_client)
        registry_name = operations._registry_name

        asset_id = f"azureml://registries/{registry_name}/{asset_type_pluralized}/" \
                   f"{asset_name}/versions/{asset_version}"
        logger.print(f"getting data reference for asset {asset_id}")
        for cnt in range(1, RegistryUtils.RETRY_COUNT + 1):
            try:
                response = RegistryUtils._get_temp_data_ref(asset_name, asset_version, asset_type, ml_client)
                blob_uri = response.blob_reference_for_consumption.blob_uri
                sas_uri = response.blob_reference_for_consumption.credential.additional_properties["sasUri"]
                if not blob_uri or not sas_uri:
                    raise Exception("Error fetching blob or SAS URI")
                return blob_uri, sas_uri
            except Exception as e:
                logger.log_error(f"Exception in fetching data reference. Try #{cnt}. Error: {e}")
        else:
            raise Exception(f"Unable to fetch data reference for asset {asset_id}")

    def _get_temp_data_ref(asset_name: str, asset_version: str, asset_type: assets.AssetType, ml_client: MLClient):
        asset_type_pluralized = RegistryUtils.pluralize_asset_type(asset_type)
        operations = RegistryUtils.get_operations_from_type(asset_type=asset_type, ml_client=ml_client)
        service_client = operations._service_client
        resource_group_name = operations._resource_group_name
        registry_name = operations._registry_name

        body = get_asset_body_for_registry_storage(registry_name, asset_type_pluralized, asset_name, asset_version)
        response = service_client.temporary_data_references.create_or_get_temporary_data_reference(
            name=asset_name,
            version=asset_version,
            resource_group_name=resource_group_name,
            registry_name=registry_name,
            body=body,
        )
        return response


class Asset:
    """Asset class."""

    def __init__(self, spec_path: Path, extra_config: Config, registry_name: str, temp_dir: Path,
                 copy_updater: CopyUpdater = None, output_level: str = "essential"):
        """Initialize asset."""
        self._spec_path = spec_path
        self._extra_config = extra_config
        self._temp_dir = temp_dir
        self._copy_updater = copy_updater
        self._output_level = output_level


class DataAsset(Asset):
    """Asset class for data."""

    def __init__(self, spec_path: Path, data_config: DataConfig, registry_name: str, temp_dir: Path,
                 copy_updater: CopyUpdater = None, output_level: str = "essential"):
        """Initialize data asset."""
        super().__init__(spec_path, data_config, registry_name, temp_dir, copy_updater, output_level)
        self._data_config = data_config

        try:
            self._data = load_data(spec_path)
        except Exception as e:
            logger.error(f"Error in loading data spec file at {spec_path}: {e}")
            return False

    def prepare_data(self, ml_client: MLClient):
        """Prepare data for publish."""
        data_registry_path = RegistryUtils.publish_to_registry(ml_client, self._data_config, self._data.name,
                                                               self._data.version, AssetType.DATA, self._temp_dir,
                                                               self._copy_updater, self._output_level)
        self._data.path = data_registry_path
        return self._data


class ModelAsset(Asset):
    """Asset class for model."""

    def __init__(self, spec_path: Path, model_config: ModelConfig, registry_name: str, temp_dir: Path,
                 copy_updater: CopyUpdater = None, output_level: str = "essential"):
        """Initialize model asset."""
        super().__init__(spec_path, model_config, registry_name, temp_dir, copy_updater, output_level)
        self._model_config = model_config

        try:
            self._model = load_model(spec_path)
            self._model.description = model_config.description
            self._model.type = model_config.type.value
            if self._model.tags:
                self._model.tags = {k: resolve_from_file_for_asset(self._model_config, v)
                                    for k, v in self._model.tags.items()}
        except Exception as e:
            logger.error(f"Error in loading model spec file at {spec_path}: {e}")
            return False


class MLFlowModelAsset(ModelAsset):
    """Asset class for MLflow model."""

    MLMODEL_FILE_NAME = "MLmodel"
    MLFLOW_MODEL_PATH = "mlflow_model_folder"

    def __init__(self, spec_path: Path, model_config: ModelConfig, registry_name: str, temp_dir: Path,
                 copy_updater: CopyUpdater = None, output_level: str = "essential"):
        """Initialize Mlflow model asset."""
        super().__init__(spec_path, model_config, registry_name, temp_dir, copy_updater, output_level)

    def prepare_model(self, ml_client: MLClient):
        """Prepare model for publish."""
        model_registry_path = RegistryUtils.publish_to_registry(ml_client, self._model_config, self._model.name,
                                                                self._model.version, AssetType.MODEL, self._temp_dir,
                                                                self._copy_updater, self._output_level)
        self._model.path = model_registry_path + "/" + MLFlowModelAsset.MLFLOW_MODEL_PATH
        return self._model


class TritonModelAsset(ModelAsset):
    """Asset class for Triton model."""

    def __init__(self, spec_path: Path, model_config: ModelConfig, registry_name: str, temp_dir: Path,
                 copy_updater: CopyUpdater = None, output_level: str = "essential"):
        """Initialize Triton model asset."""
        super().__init__(spec_path, model_config, registry_name, temp_dir, copy_updater, output_level)

    def prepare_model(self, ml_client: MLClient):
        """Prepare model for publish."""
        model_registry_path = RegistryUtils.publish_to_registry(ml_client, self._model_config, self._model.name,
                                                                self._model.version, AssetType.MODEL, self._temp_dir,
                                                                self._copy_updater, self._output_level)
        self._model.path = model_registry_path
        return self._model


class CustomModelAsset(ModelAsset):
    """Asset class for custom model."""

    def __init__(self, spec_path: Path, model_config: ModelConfig, registry_name: str, temp_dir: Path,
                 copy_updater: CopyUpdater = None, output_level: str = "essential"):
        """Initialize custom model asset."""
        super().__init__(spec_path, model_config, registry_name, temp_dir, copy_updater, output_level)

    def prepare_model(self, ml_client: MLClient):
        """Prepare model for publish."""
        model_registry_path = RegistryUtils.publish_to_registry(ml_client, self._model_config, self._model.name,
                                                                self._model.version, AssetType.MODEL, self._temp_dir,
                                                                self._copy_updater, self._output_level)
        self._model.path = model_registry_path
        return self._model


def prepare_model(spec_path: Path, model_config: ModelConfig, temp_dir: Path, ml_client: MLClient,
                  copy_updater: CopyUpdater = None, output_level: str = "essential"):
    """Prepare model for publish."""
    try:
        logger.print(f"Model type: {model_config.type}")
        registry_name = ml_client.models._registry_name
        if model_config.type == assets.ModelType.CUSTOM:
            model_asset = CustomModelAsset(spec_path, model_config, registry_name, temp_dir,
                                           copy_updater, output_level)
        elif model_config.type == assets.ModelType.MLFLOW:
            model_asset = MLFlowModelAsset(spec_path, model_config, registry_name, temp_dir,
                                           copy_updater, output_level)
        elif model_config.type == assets.ModelType.TRITON:
            model_asset = TritonModelAsset(spec_path, model_config, registry_name, temp_dir,
                                           copy_updater, output_level)
        else:
            logger.log_error(f"Model type {model_config.type.value} not supported")
            return False
        model = model_asset.prepare_model(ml_client)
        return model, True
    except Exception as e:
        logger.log_error(f"prepare model failed for {spec_path}. Error {e}")
        return None, False


def prepare_data(spec_path: Path, data_config: DataConfig, temp_dir: Path, ml_client: MLClient,
                 copy_updater: CopyUpdater = None, output_level: str = "essential"):
    """Prepare data for publish."""
    try:
        registry_name = ml_client.data._registry_name
        data_asset = DataAsset(spec_path, data_config, registry_name, temp_dir, copy_updater, output_level)
        data = data_asset.prepare_data(ml_client)
        return data, True
    except Exception as e:
        logger.log_error(f"prepare data failed for {spec_path}. Error {e}")
        return None, False


def update_metadata(
    name: str,
    version: str,
    update: AssetVersionUpdate,
    ml_client: MLClient,
    asset_type: assets.AssetType,
    allow_no_op_update: bool = False,
):
    """Update the mutable metadata of an already registered asset."""
    try:
        # Get existing asset
        operations = RegistryUtils.get_operations_from_type(asset_type=asset_type, ml_client=ml_client)
        asset = operations.get(name=name, version=version)
        need_update = False

        # Update tags
        if update.tags:
            updated_tags = copy.deepcopy(asset.tags)

            if update.tags.replace is not None:
                updated_tags = update.tags.replace
            else:
                if update.tags.add is not None:
                    for k, v in update.tags.add.items():
                        updated_tags[k] = v
                if update.tags.delete is not None:
                    for k in update.tags.delete:
                        updated_tags.pop(k, None)

            if updated_tags != asset.tags:
                logger.print("tags has been updated.")
                asset.tags = updated_tags
                need_update = True

        # Update system_metadata
        if update.system_metadata and hasattr(asset, '_system_metadata'):
            updated_system_metadata = copy.deepcopy(asset._system_metadata)

            if update.system_metadata.replace is not None:
                updated_system_metadata = update.system_metadata.replace
            else:
                if update.system_metadata.add is not None:
                    for k, v in update.system_metadata.add.items():
                        updated_system_metadata[k] = v
                if update.system_metadata.delete is not None:
                    for k in update.system_metadata.delete:
                        updated_system_metadata.pop(k, None)

            if updated_system_metadata != asset._system_metadata:
                logger.print("system_metadata has been updated.")
                asset._system_metadata = updated_system_metadata
                need_update = True

        # Update properties
        if update.properties:
            updated_properties = copy.deepcopy(asset.properties)
            if update.properties.add is not None:
                for k, v in update.properties.add.items():
                    if k in asset.properties and asset.properties[k] != v:
                        raise Exception(f"Value of property {k} for {asset_type.value} {name} cannot "
                                        f"be replaced to {v} without increasing the version.")
                    updated_properties[k] = v

            if updated_properties != asset.properties:
                logger.print("properties has been updated")
                asset.properties = updated_properties
                need_update = True

        # Update description
        if update.description is not None and asset.description != update.description:
            logger.print("description has been updated")
            asset.description = update.description
            need_update = True

        if allow_no_op_update or need_update:
            operations.create_or_update(asset)
            logger.print(f"{asset_type.value.capitalize()} metadata updated successfully for {name}")
        else:
            logger.print(f"No update found for {asset_type.value} {name}. Skipping")
    except Exception as e:
        logger.log_error(f"Failed to update metadata for {asset_type.value} : {name} : {e}")

    # Archive or restore asset based on stage
    try:
        if update.stage == "Archived":
            logger.print(f"Archiving {asset_type.value} {name} version {version}")
            operations.archive(name=name, version=version)
        elif update.stage == "Active":
            logger.print(f"Restoring {asset_type.value} {name} version {version}")
            operations.restore(name=name, version=version)
    except Exception as e:
        logger.log_error(f"Failed to archive or restore {asset_type.value} {name} version {version}: {e}")

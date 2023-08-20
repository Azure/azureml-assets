# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Model utils Operations Class."""

import os
import azureml.assets as assets
from azure.identity import AzureCliCredential
from azure.ai.ml import load_data
from azure.ai.ml._utils._registry_utils import get_asset_body_for_registry_storage, get_registry_client
from azureml.assets.util import logger
from azureml.assets.config import PathType
from azureml.assets.model.download_utils import copy_azure_artifacts
from pathlib import Path


class RegistryUtils:
    """Registry utils."""

    RETRY_COUNT = 3

    def get_registry_data_reference(registry_name: str, data_name: str, data_version: str):
        """Fetch data reference for asset in the registry."""
        asset_id = f"azureml://registries/{registry_name}/data/{data_name}/versions/{data_version}"
        logger.print(f"getting data reference for asset {asset_id}")
        for cnt in range(1, RegistryUtils.RETRY_COUNT + 1):
            try:
                response = RegistryUtils._get_temp_data_ref(registry_name, data_name, data_version)
                blob_uri = response.blob_reference_for_consumption.blob_uri
                sas_uri = response.blob_reference_for_consumption.credential.additional_properties["sasUri"]
                if not blob_uri or not sas_uri:
                    raise Exception("Error in fetching BLOB or SAS URI")
                return blob_uri, sas_uri
            except Exception as e:
                logger.log_error(f"Exception in fetching data reference. Try #{cnt}. Error: {e}")
        else:
            raise Exception(f"Unable to fetch data reference for asset {asset_id}")

    def _get_temp_data_ref(registry_name, data_name, data_version):
        try:
            credential = AzureCliCredential()
            body = get_asset_body_for_registry_storage(registry_name, "data", data_name, data_version)
            registry_client, resource_group, _ = get_registry_client(credential, registry_name)
            response = registry_client.temporary_data_references.create_or_get_temporary_data_reference(
                name=data_name,
                version=data_version,
                resource_group_name=resource_group,
                registry_name=registry_name,
                body=body,
            )
            return response
        except Exception as e:
            logger.log_error(f"exception in fetching data reference: {e}")


class DataAsset:
    """Asset class for model."""

    def __init__(self, spec_path, data_config, registry_name, temp_dir):
        """Initialize model asset."""
        self._spec_path = spec_path
        self._data_config = data_config
        self._registry_name = registry_name
        self._temp_dir = temp_dir

        try:
            self._data = load_data(spec_path)
            self._data.type = data_config.type.value
        except Exception as e:
            logger.error(f"Error in loading data spec file at {spec_path}: {e}")
            return False

        data_description_file_path = Path(spec_path).parent / data_config.description
        logger.print(f"data_description_file_path {data_description_file_path}")
        if os.path.exists(data_description_file_path):
            with open(data_description_file_path) as f:
                data_description = f.read()
                self._data.description = data_description
        else:
            logger.print("description file does not exist")
        

    def _publish_to_registry(self):
        src_uri = self._data_config.path.uri

        try:
            # copy to registry blobstorage
            logger.print("get data ref for registry storage upload")
            blob_uri, sas_uri = RegistryUtils.get_registry_data_reference(
                self._registry_name, self._model.name, self._model.version
            )
            success = copy_azure_artifacts(src_uri, sas_uri)
            if not success:
                raise Exception("blobstorage copy failed.")
            logger.print("Successfully copied data artifacts to registry storage")
            return blob_uri
        except Exception as e:
            raise Exception(f"Error in copying artifacts to registry storage. Error {e}")


class CustomDataAsset(DataAsset):
    """Asset class for custom data."""

    def __init__(self, spec_path, data_config, registry_name, temp_dir):
        """Initialize custom model asset."""
        super().__init__(spec_path, data_config, registry_name, temp_dir)

    def prepare_model(self):
        """Prepare model for publish."""
        data_registry_path = self._publish_to_registry()
        self._data.path = data_registry_path
        return self._data


def prepare_data(spec_path, data_config, registry_name, temp_dir):
    """Prepare data for publish."""
    try:
        logger.print(f"Data type: {data_config.type}")
        if data_config.type == assets.DataAssetType.URI_FILE or data_config.type == assets.DataAssetType.URI_FOLDER:
            data_asset = CustomDataAsset(spec_path, data_config, registry_name, temp_dir)
        else:
            logger.log_error(f"Data type {data_config.type.value} not supported")
            return False
        data = data_asset.prepare_data()
        return data, True
    except Exception as e:
        logger.log_error(f"prepare data failed for {spec_path}. Error {e}")
        return None, False

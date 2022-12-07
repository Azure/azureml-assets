# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import logging
import pytest
from .test_utilities import (
    load_json,
    make_request,
    validate_successful_run,
    register_data_assets,
    update_payload_with_registered_data_assets,
    update_payload_module_id,
)
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities._load_functions import load_component


logger = logging.getLogger(name=__file__)


@pytest.mark.unittest
class TestAutoMLClassificationComponent:
    def register_classification_assets(self, mlclient, version):
        classification_data_assets = (
            {
                "name": "classification_training_data",
                "path": "./training/automl/tests/test_configs/assets/classification_bankmarketing/training-mltable-folder",
                "type": AssetTypes.MLTABLE,
                "version": version,
            },
            {
                "name": "classification_validation_data",
                "path": "./training/automl/tests/test_configs/assets/classification_bankmarketing/validation-mltable-folder",
                "type": AssetTypes.MLTABLE,
                "version": version,
            },
            {
                "name": "classification_test_data",
                "path": "./training/automl/tests/test_configs/assets/classification_bankmarketing/test-mltable-folder",
                "type": AssetTypes.MLTABLE,
                "version": version,
            },
        )
        assets = register_data_assets(mlclient, classification_data_assets)
        return assets

    def test_something(
        self,
        mlclient,
        registry_name,
        http_headers,
        ui_service_endpoint,
        workspace_id,
        workspace_location,
    ):
        logger.warning("test_something")
        payload_path = "training/automl/tests/test_configs/payload/canary/classification_ui_payload.json"
        spec_path = "training/automl/components/automl_tabular_classification/spec.yaml"
        classification_component = load_component(spec_path)
        version_suffix = "test01"
        version = "1"
        version = version + "-" + version_suffix

        data_assets = self.register_classification_assets(mlclient, version=version)
        payload = load_json(payload_path)
        payload = update_payload_with_registered_data_assets(
            data_assets,
            payload,
            workspace_id=workspace_id,
            workspace_location=workspace_location,
        )

        # component_asset_id = "azureml://registries/azureml-staging/components/{}/versions/{}"
        # component_asset_id.format(registry_name, classification_component.name, version)
        # temp change check staging registry
        component_asset_id = "azureml://registries/azureml-staging/components/microsoft_azureml_automl_classification_component/versions/0.0.1-preview-azureml-staging.1669926498"
        payload = update_payload_module_id(payload, "automl_node", component_asset_id)

        status, pipeline_run_id = make_request(
            uri=ui_service_endpoint, method="POST", headers=http_headers, data=payload
        )
        assert status == 200

        logger.info("pipeline_run_id : " + str(pipeline_run_id))
        validate_successful_run(mlclient, pipeline_run_id)

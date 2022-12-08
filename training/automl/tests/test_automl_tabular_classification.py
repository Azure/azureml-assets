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
    def register_assets(self, mlclient, version):
        data_assets = (
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
        assets = register_data_assets(mlclient, data_assets)
        return assets

    @pytest.mark.parametrize(
        "spec_path, payload_path", [
            (
                "training/automl/components/automl_tabular_classification/spec.yaml",
                "training/automl/tests/test_configs/payload/canary/classification_ui_payload.json"
            )
        ]
    )
    def test_classification(
        self,
        mlclient,
        registry_name,
        http_headers,
        ui_service_endpoint,
        workspace_id,
        workspace_location,
        version_suffix,
        spec_path,
        payload_path,
    ):
        component = load_component(spec_path)
        version = component.version + "-" + version_suffix if version_suffix else component.version

        data_assets = self.register_assets(mlclient, version="1")
        payload = load_json(payload_path)
        payload = update_payload_with_registered_data_assets(
            data_assets,
            payload,
            workspace_id=workspace_id,
            workspace_location=workspace_location,
        )

        COMPONENT_ASSET_TEMPLATE = "azureml://registries/{}/components/{}/versions/{}"
        component_asset_id = COMPONENT_ASSET_TEMPLATE.format(
            registry_name,
            component.name,
            version,
        )
        logger.info("component_asset_id => " + component_asset_id)
        payload = update_payload_module_id(payload, "automl_node", component_asset_id)
        status, pipeline_run_id = make_request(
            uri=ui_service_endpoint, method="POST", headers=http_headers, data=payload
        )
        assert status == 200
        logger.info("pipeline_run_id : " + str(pipeline_run_id))
        validate_successful_run(mlclient, pipeline_run_id)

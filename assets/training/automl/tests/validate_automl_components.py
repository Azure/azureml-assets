# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Test AutoML designer components."""

import logging
import pytest
from .utils import (
    load_json,
    make_request,
    validate_successful_run,
    register_data_assets,
    update_payload_with_registered_data_assets,
    update_payload_module_id,
)


logger = logging.getLogger(name=__file__)

AUTOML_NODE = "automl_node"
COMPONENT_ASSET_DEFAULT_LABEL_TEMPLATE = "azureml://registries/{}/components/{}/labels/default"
COMPONENT_ASSET_WITH_VERSION_TEMPLATE = "azureml://registries/{}/components/{}/versions/{}"


@pytest.mark.unittest
class ValidateAutoMLComponents:
    """ValidateAutoMLComponents."""

    def validate_automl_components(
        self,
        mlclient,
        component_name,
        payload_path,
        data_assets,
        registry_name,
        http_headers,
        ui_service_endpoint,
        workspace_id,
        workspace_location,
    ):
        """Test AutoML designer components."""
        component_asset_id = COMPONENT_ASSET_DEFAULT_LABEL_TEMPLATE.format(registry_name, component_name)
        logger.info(f"component_asset_id => {component_asset_id}")

        data_assets = register_data_assets(mlclient, data_assets)
        logger.info("registered data assets")

        payload = load_json(payload_path)
        payload = update_payload_module_id(payload, AUTOML_NODE, component_asset_id)
        payload = update_payload_with_registered_data_assets(
            data_assets,
            payload,
            workspace_id=workspace_id,
            workspace_location=workspace_location,
        )

        status, pipeline_run_id = make_request(
            uri=ui_service_endpoint, method="POST", headers=http_headers, data=payload
        )
        assert status == 200
        logger.info(f"pipeline_run_id : {pipeline_run_id}")
        validate_successful_run(mlclient, pipeline_run_id)

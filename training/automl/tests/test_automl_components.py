# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Test AutoML designer components."""

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

AUTOML_NODE = "automl_node"
COMPONENT_ASSET_DEFAULT_LABEL_TEMPLATE = "azureml://registries/{}/components/{}/labels/default"
COMPONENT_ASSET_WITH_VERSION_TEMPLATE = "azureml://registries/{}/components/{}/versions/{}"


@pytest.mark.unittest
class TestAutoMLComponents:
    """TestAutoMLComponents."""

    @pytest.mark.parametrize(
        "spec_path, payload_path, data_assets",
        [
            # classification
            (
                "automl/components/automl_tabular_classification/spec.yaml",
                "automl/tests/test_configs/payload/classification_bankmarketing_payload.json",
                [
                    {
                        "name": "classification_training_data",
                        "path": (
                            "./automl/tests/test_configs/assets/classification-bankmarketing/training-mltable-folder"
                        ),
                        "type": AssetTypes.MLTABLE,
                        "version": "1",
                    },
                    {
                        "name": "classification_validation_data",
                        "path": (
                            "./automl/tests/test_configs/assets/classification-bankmarketing/validation-mltable-folder"
                        ),
                        "type": AssetTypes.MLTABLE,
                        "version": "1",
                    },
                    {
                        "name": "classification_test_data",
                        "path": "./automl/tests/test_configs/assets/classification-bankmarketing/test-mltable-folder",
                        "type": AssetTypes.MLTABLE,
                        "version": "1",
                    },
                ],
            ),
            # regression
            (
                "automl/components/automl_tabular_regression/spec.yaml",
                "automl/tests/test_configs/payload/regression_hardware_performance_payload.json",
                [
                    {
                        "name": "regression_training_data",
                        "path": (
                            "./automl/tests/test_configs/assets/regression-hardware-performance/"
                            + "training-mltable-folder"
                        ),
                        "type": AssetTypes.MLTABLE,
                        "version": "1",
                    }
                ],
            ),
            # forecasting
            (
                "automl/components/automl_tabular_forecasting/spec.yaml",
                "automl/tests/test_configs/payload/forecasting_energy_demand_payload.json",
                [
                    {
                        "name": "forecasting_training_data",
                        "path": (
                            "./automl/tests/test_configs/assets/forecasting-energy-demand/training-mltable-folder"
                        ),
                        "type": AssetTypes.MLTABLE,
                        "version": "1",
                    },
                ],
            ),
            # nlp classification
            (
                "automl/components/automl_text_classification/spec.yaml",
                "automl/tests/test_configs/payload/text_classification_newsgroup_payload.json",
                [
                    {
                        "name": "text_classification_training_data",
                        "path": (
                            "./automl/tests/test_configs/assets/text-classification-newsgroup/"
                            + "training-mltable-folder"
                        ),
                        "type": AssetTypes.MLTABLE,
                        "version": "1",
                    },
                    {
                        "name": "text_classification_validation_data",
                        "path": (
                            "./automl/tests/test_configs/assets/text-classification-newsgroup/"
                            + "validation-mltable-folder"
                        ),
                        "type": AssetTypes.MLTABLE,
                        "version": "1",
                    },
                ],
            ),
            # nlp classification multilable
            (
                "automl/components/automl_text_classification_multilable/spec.yaml",
                "automl/tests/test_configs/payload/text_classification_multilable_newsgroup_payload.json",
                [
                    {
                        "name": "text_classification_multilable_training_data",
                        "path": (
                            "./automl/tests/test_configs/assets/text-classification-multilabel-paper-categorization/"
                            + "training-mltable-folder"
                        ),
                        "type": AssetTypes.MLTABLE,
                        "version": "1",
                    },
                    {
                        "name": "text_classification_multilable_validation_data",
                        "path": (
                            "./automl/tests/test_configs/assets/text-classification-multilabel-paper-categorization/"
                            + "validation-mltable-folder"
                        ),
                        "type": AssetTypes.MLTABLE,
                        "version": "1",
                    },
                ],
            ),
            # nlp ner
            (
                "automl/components/automl_text_ner/spec.yaml",
                "automl/tests/test_configs/payload/text_ner_conll_payload.json",
                [
                    {
                        "name": "text_ner_training_data",
                        "path": "./automl/tests/test_configs/assets/text-ner-conll/training-mltable-folder",
                        "type": AssetTypes.MLTABLE,
                        "version": "1",
                    },
                    {
                        "name": "text_ner_validation_data",
                        "path": "./automl/tests/test_configs/assets/text-ner-conll/validation-mltable-folder",
                        "type": AssetTypes.MLTABLE,
                        "version": "1",
                    },
                ],
            ),
            # image classification
            (
                "automl/components/automl_image_classification/spec.yaml",
                "automl/tests/test_configs/payload/image_classification_fridge_items_payload.json",
                [
                    {
                        "name": "image_classification_training_data",
                        "path": "./automl/tests/test_configs/assets/image-classification-fridge-items/training-mltable-folder",
                        "type": AssetTypes.MLTABLE,
                        "version": "1",
                    },
                    {
                        "name": "image_classification_validation_data",
                        "path": "./automl/tests/test_configs/assets/image-classification-fridge-items/validation-mltable-folder",
                        "type": AssetTypes.MLTABLE,
                        "version": "1",
                    },
                ]
            ),
            # image classification multilabel
            (
                "automl/components/automl_image_classification_multilabel/spec.yaml",
                "automl/tests/test_configs/payload/image_classification_multilabel_fridge_items_payload.json",
                [
                    {
                        "name": "image_classification_multilabel_training_data",
                        "path": "./automl/tests/test_configs/assets/image-classification-multilabel-fridge-items/training-mltable-folder",
                        "type": AssetTypes.MLTABLE,
                        "version": "1",
                    },
                    {
                        "name": "image_classification_multilabel_validation_data",
                        "path": "./automl/tests/test_configs/assets/image-classification-multilabel-fridge-items/validation-mltable-folder",
                        "type": AssetTypes.MLTABLE,
                        "version": "1",
                    },
                ]
            ),
            # image object detection
            (
                "automl/components/automl_image_object_detection/spec.yaml",
                "automl/tests/test_configs/payload/image_object_detection_fridge_items_payload.json",
                [
                    {
                        "name": "image_object_detection_training_data",
                        "path": "./automl/tests/test_configs/assets/image-object-detection-fridge-items/training-mltable-folder",
                        "type": AssetTypes.MLTABLE,
                        "version": "1",
                    },
                    {
                        "name": "image_object_detection_validation_data",
                        "path": "./automl/tests/test_configs/assets/image-object-detection-fridge-items/validation-mltable-folder",
                        "type": AssetTypes.MLTABLE,
                        "version": "1",
                    },
                ]
            ),
            # image instance segmentation
            (
                "automl/components/automl_image_instance_segmentation/spec.yaml",
                "automl/tests/test_configs/payload/image_instance_segmentation_fridge_items_payload.json",
                [
                    {
                        "name": "image_instance_segmentation_training_data",
                        "path": "./automl/tests/test_configs/assets/image-instance-segmentation-fridge-items/training-mltable-folder",
                        "type": AssetTypes.MLTABLE,
                        "version": "1",
                    },
                    {
                        "name": "image_instance_segmentation_validation_data",
                        "path": "./automl/tests/test_configs/assets/image-instance-segmentation-fridge-items/validation-mltable-folder",
                        "type": AssetTypes.MLTABLE,
                        "version": "1",
                    },
                ]
            ),
        ],
    )
    def test_automl_components(
        self,
        mlclient,
        spec_path,
        payload_path,
        data_assets,
        registry_name,
        http_headers,
        ui_service_endpoint,
        workspace_id,
        workspace_location,
    ):
        """Test AutoML designer components."""
        component = load_component(spec_path)
        component_asset_id = COMPONENT_ASSET_DEFAULT_LABEL_TEMPLATE.format(registry_name, component.name)
        logger.info("component_asset_id => " + component_asset_id)

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
        logger.info("pipeline_run_id : " + str(pipeline_run_id))
        validate_successful_run(mlclient, pipeline_run_id)

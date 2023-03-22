# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Test AutoML Vision designer components."""

import logging
import pytest
from .validate_automl_components import ValidateAutoMLComponents
from azure.ai.ml.constants import AssetTypes


logger = logging.getLogger(name=__file__)


@pytest.mark.unittest
class TestAutoMLVisionComponents(ValidateAutoMLComponents):
    """TestAutoMLVisionComponents."""

    @pytest.mark.parametrize(
        "spec_path, payload_path, data_assets",
        [
            # image classification
            (
                "automl_image_classification",
                "automl/tests/test_configs/payload/image_classification_fridge_items_payload.json",
                [
                    {
                        "name": "image_classification_training_data",
                        "path": (
                            "./automl/tests/test_configs/assets/image-classification-fridge-items/"
                            + "training-mltable-folder"
                        ),
                        "type": AssetTypes.MLTABLE,
                        "version": "0.0.1",
                    },
                    {
                        "name": "image_classification_validation_data",
                        "path": (
                            "./automl/tests/test_configs/assets/image-classification-fridge-items/"
                            + "validation-mltable-folder"
                        ),
                        "type": AssetTypes.MLTABLE,
                        "version": "0.0.1",
                    },
                ],
            ),
            # image classification multilabel
            (
                "automl_image_classification_multilabel",
                "automl/tests/test_configs/payload/image_classification_multilabel_fridge_items_payload.json",
                [
                    {
                        "name": "image_classification_multilabel_training_data",
                        "path": (
                            "./automl/tests/test_configs/assets/image-classification-multilabel-fridge-items"
                            + "/training-mltable-folder"
                        ),
                        "type": AssetTypes.MLTABLE,
                        "version": "0.0.1",
                    },
                    {
                        "name": "image_classification_multilabel_validation_data",
                        "path": (
                            "./automl/tests/test_configs/assets/image-classification-multilabel-fridge-items/"
                            + "validation-mltable-folder"
                        ),
                        "type": AssetTypes.MLTABLE,
                        "version": "0.0.1",
                    },
                ],
            ),
            # image object detection
            (
                "automl_image_object_detection",
                "automl/tests/test_configs/payload/image_object_detection_fridge_items_payload.json",
                [
                    {
                        "name": "image_object_detection_training_data",
                        "path": (
                            "./automl/tests/test_configs/assets/image-object-detection-fridge-items/"
                            + "training-mltable-folder"
                        ),
                        "type": AssetTypes.MLTABLE,
                        "version": "0.0.1",
                    },
                    {
                        "name": "image_object_detection_validation_data",
                        "path": (
                            "./automl/tests/test_configs/assets/image-object-detection-fridge-items/"
                            + "validation-mltable-folder"
                        ),
                        "type": AssetTypes.MLTABLE,
                        "version": "0.0.1",
                    },
                ],
            ),
            # image instance segmentation
            (
                "automl_image_instance_segmentation",
                "automl/tests/test_configs/payload/image_instance_segmentation_fridge_items_payload.json",
                [
                    {
                        "name": "image_instance_segmentation_training_data",
                        "path": (
                            "./automl/tests/test_configs/assets/image-instance-segmentation-fridge-items/"
                            + "training-mltable-folder"
                        ),
                        "type": AssetTypes.MLTABLE,
                        "version": "0.0.1",
                    },
                    {
                        "name": "image_instance_segmentation_validation_data",
                        "path": (
                            "./automl/tests/test_configs/assets/image-instance-segmentation-fridge-items/"
                            + "validation-mltable-folder"
                        ),
                        "type": AssetTypes.MLTABLE,
                        "version": "0.0.1",
                    },
                ],
            ),
        ],
    )
    def test_automl_vision_components(
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
        """Test AutoML Vision components."""
        logger.info("Running AutoML Vision Component Validations ...")
        super().validate_automl_components(
            mlclient,
            spec_path,
            payload_path,
            data_assets,
            registry_name,
            http_headers,
            ui_service_endpoint,
            workspace_id,
            workspace_location,
        )

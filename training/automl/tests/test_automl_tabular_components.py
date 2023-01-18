# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Test AutoML Tabular designer components."""

import logging
import pytest
from .validate_automl_components import ValidateAutoMLComponents
from azure.ai.ml.constants import AssetTypes


logger = logging.getLogger(name=__file__)


@pytest.mark.unittest
class TestAutoMLTabularComponents(ValidateAutoMLComponents):
    """TestAutoMLTabularComponents."""

    @pytest.mark.parametrize(
        "spec_path, payload_path, data_assets",
        [
            # classification
            (
                "automl_classification",
                "automl/tests/test_configs/payload/classification_bankmarketing_payload.json",
                [
                    {
                        "name": "classification_training_data",
                        "path": (
                            "./automl/tests/test_configs/assets/classification-bankmarketing/training-mltable-folder"
                        ),
                        "type": AssetTypes.MLTABLE,
                        "version": "0.0.1",
                    },
                    {
                        "name": "classification_validation_data",
                        "path": (
                            "./automl/tests/test_configs/assets/classification-bankmarketing/validation-mltable-folder"
                        ),
                        "type": AssetTypes.MLTABLE,
                        "version": "0.0.1",
                    },
                    {
                        "name": "classification_test_data",
                        "path": "./automl/tests/test_configs/assets/classification-bankmarketing/test-mltable-folder",
                        "type": AssetTypes.MLTABLE,
                        "version": "0.0.1",
                    },
                ],
            ),
            # regression
            (
                "automl_regression",
                "automl/tests/test_configs/payload/regression_hardware_performance_payload.json",
                [
                    {
                        "name": "regression_training_data",
                        "path": (
                            "./automl/tests/test_configs/assets/regression-hardware-performance/"
                            + "training-mltable-folder"
                        ),
                        "type": AssetTypes.MLTABLE,
                        "version": "0.0.1",
                    }
                ],
            ),
            # forecasting
            (
                "automl_forecasting",
                "automl/tests/test_configs/payload/forecasting_energy_demand_payload.json",
                [
                    {
                        "name": "forecasting_training_data",
                        "path": (
                            "./automl/tests/test_configs/assets/forecasting-energy-demand/training-mltable-folder"
                        ),
                        "type": AssetTypes.MLTABLE,
                        "version": "0.0.1",
                    },
                ],
            ),
        ],
    )
    def test_automl_tabular_components(
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
        """Test AutoML Tabular components."""
        logger.info("Running AutoML Tabular Component Validations ...")
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

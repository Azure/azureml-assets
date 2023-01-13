# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Test AutoML NLP designer components."""

import logging
import pytest
from .validate_automl_components import ValidateAutoMLComponents
from azure.ai.ml.constants import AssetTypes


logger = logging.getLogger(name=__file__)


@pytest.mark.unittest
class TestNLPComponents(ValidateAutoMLComponents):
    """TestNLPComponents."""

    @pytest.mark.parametrize(
        "spec_path, payload_path, data_assets",
        [
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
                        "version": "0.0.1",
                    },
                    {
                        "name": "text_classification_validation_data",
                        "path": (
                            "./automl/tests/test_configs/assets/text-classification-newsgroup/"
                            + "validation-mltable-folder"
                        ),
                        "type": AssetTypes.MLTABLE,
                        "version": "0.0.1",
                    },
                ],
            ),
            # nlp classification multilabel
            (
                "automl/components/automl_text_classification_multilabel/spec.yaml",
                "automl/tests/test_configs/payload/text_classification_multilabel_newsgroup_payload.json",
                [
                    {
                        "name": "text_classification_multilabel_training_data",
                        "path": (
                            "./automl/tests/test_configs/assets/text-classification-multilabel-paper-categorization/"
                            + "training-mltable-folder"
                        ),
                        "type": AssetTypes.MLTABLE,
                        "version": "0.0.1",
                    },
                    {
                        "name": "text_classification_multilabel_validation_data",
                        "path": (
                            "./automl/tests/test_configs/assets/text-classification-multilabel-paper-categorization/"
                            + "validation-mltable-folder"
                        ),
                        "type": AssetTypes.MLTABLE,
                        "version": "0.0.1",
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
                        "version": "0.0.1",
                    },
                    {
                        "name": "text_ner_validation_data",
                        "path": "./automl/tests/test_configs/assets/text-ner-conll/validation-mltable-folder",
                        "type": AssetTypes.MLTABLE,
                        "version": "0.0.1",
                    },
                ],
            ),
        ],
    )
    def test_nlp_components(
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
        """Test AutoML NLP components."""
        logger.info("Running NLP Component Validations ...")
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

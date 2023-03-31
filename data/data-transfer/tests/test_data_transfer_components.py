# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Test AutoML NLP designer components."""

import logging
import pytest
from azure.ai.ml import load_job
from azure.ai.ml.entities import PipelineJob


logger = logging.getLogger(name=__file__)


@pytest.mark.unittest
class TestDataTransferComponents:
    """TestDataTransferComponents."""

    def test_pipeline_job_load_with_data_transfer_components(self):
        """Test DataTransfer components."""
        logger.info("Running DataTransfer Component Validations ...")
        pipeline = load_job(source="./data-transfer/tests/test_data_transfer_components.yaml")
        assert isinstance(pipeline, PipelineJob)

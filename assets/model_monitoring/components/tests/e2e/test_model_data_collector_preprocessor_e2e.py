# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains e2e tests for the model data collector preprocessor component."""

import pytest
from azure.ai.ml import Input, MLClient, Output
from azure.ai.ml.entities import Spark, AmlTokenConfiguration
from azure.ai.ml.dsl import pipeline
from tests.e2e.utils.constants import (
    COMPONENT_NAME_MDC_PREPROCESSOR,
    DATA_ASSET_IRIS_MODEL_INPUTS_WITH_DRIFT
)

testdata = [('True'), ('False')]


def _submit_mdc_preprocessor_job(
    ml_client: MLClient,
    get_component,
    experiment_name,
    extract_correlation_id,
    input_data
):
    mdc_preprocessor_component = get_component(COMPONENT_NAME_MDC_PREPROCESSOR)

    @pipeline
    def _mdc_preprocessor_e2e():
        mdc_preprocessor_output: Spark = mdc_preprocessor_component(
            data_window_start="2023-01-29T00:00:00Z",
            data_window_end="2023-02-03T00:00:00Z",
            input_data=Input(path=input_data, mode="direct", type="uri_folder"),
            extract_correlation_id=extract_correlation_id
        )

        mdc_preprocessor_output.identity = AmlTokenConfiguration()
        mdc_preprocessor_output.resources = {
            'instance_type': 'Standard_E8S_V3',
            'runtime_version': '3.3',
        }

        return {
            'preprocessed_input_data': mdc_preprocessor_output.outputs.preprocessed_input_data
        }

    pipeline_job = _mdc_preprocessor_e2e()
    pipeline_job.outputs.preprocessed_input_data = Output(
        type='mltable', mode='direct'
    )

    pipeline_job = ml_client.jobs.create_or_update(
        pipeline_job, experiment_name=experiment_name
    )

    # Wait until the job completes
    ml_client.jobs.stream(pipeline_job.name)
    return ml_client.jobs.get(pipeline_job.name)


@pytest.mark.e2e
class TestMDCPreprocessorE2E:
    """Test class."""

    @pytest.mark.parametrize("extract_correlation_id", testdata)
    def test_mdc_preprocessor_successful(
        self, ml_client: MLClient, get_component, test_suite_name, extract_correlation_id
    ):
        """Test the happy path scenario for MDC preprocessor."""
        pipeline_job = _submit_mdc_preprocessor_job(
            ml_client,
            get_component,
            test_suite_name,
            extract_correlation_id,
            DATA_ASSET_IRIS_MODEL_INPUTS_WITH_DRIFT
        )

        assert pipeline_job.status == 'Completed'

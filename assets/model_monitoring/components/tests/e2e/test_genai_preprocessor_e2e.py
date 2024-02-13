# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains e2e tests for the Gen AI raw logs model data collector preprocessor component."""

import pytest
from azure.ai.ml import Input, MLClient, Output
from azure.ai.ml.entities import Spark, AmlTokenConfiguration
from azure.ai.ml.dsl import pipeline
from tests.e2e.utils.constants import (
    COMPONENT_NAME_GENAI_PREPROCESSOR,
    DATA_ASSET_GENAI_RAW_LOG_MODEL_INPUTS,
    DATA_ASSET_LLM_INPUTS
)


def _submit_genai_preprocessor_job(
    ml_client: MLClient, get_component, experiment_name,
    input_data, start_time, end_time
):
    genai_preprocessor_component = get_component(COMPONENT_NAME_GENAI_PREPROCESSOR)

    @pipeline
    def _genai_preprocessor_e2e():
        genai_preprocessor_output: Spark = genai_preprocessor_component(
            data_window_start=start_time,
            data_window_end=end_time,
            input_data=Input(path=input_data, mode="direct", type="uri_folder"),
        )

        genai_preprocessor_output.identity = AmlTokenConfiguration()
        genai_preprocessor_output.resources = {
            'instance_type': 'Standard_E4S_V3',
            'runtime_version': '3.3',
        }

        return {
            'preprocessed_span_data': genai_preprocessor_output.outputs.preprocessed_span_data,
            'aggregated_trace_data': genai_preprocessor_output.outputs.aggregated_trace_data
        }

    pipeline_job = _genai_preprocessor_e2e()
    pipeline_job.outputs.preprocessed_span_data = Output(
        type='mltable', mode='direct'
    )
    pipeline_job.outputs.aggregated_trace_data = Output(
        type='mltable', mode='direct'
    )

    pipeline_job = ml_client.jobs.create_or_update(
        pipeline_job, experiment_name=experiment_name, skip_validation=True
    )

    # Wait until the job completes
    ml_client.jobs.stream(pipeline_job.name)
    return ml_client.jobs.get(pipeline_job.name)


@pytest.mark.e2e
class TestGenAIPreprocessorE2E:
    """Test class."""

    @pytest.mark.parametrize(
        "input_data, start_time, end_time",
        [
            # traditional model
            (DATA_ASSET_GENAI_RAW_LOG_MODEL_INPUTS, "2024-02-05T00:00:00Z", "2024-02-08T00:00:00Z"),
            # LLM model, disable until we generate the LLM logs in the gated test workspace
            # (DATA_ASSET_LLM_INPUTS, "2023-10-24T22:00:00Z", "2023-10-24T23:00:00Z")
        ]
    )
    def test_mdc_preprocessor_successful(
        self, ml_client: MLClient, get_component, test_suite_name, input_data, start_time, end_time
    ):
        """Test the happy path scenario for Gen AI preprocessor."""
        pipeline_job = _submit_genai_preprocessor_job(
            ml_client,
            get_component,
            test_suite_name,
            input_data,
            start_time,
            end_time
        )

        assert pipeline_job.status == 'Completed'

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains e2e tests for the data joiner component."""

import pytest
from azure.ai.ml import Input, MLClient, Output
from azure.ai.ml.entities import Spark, AmlTokenConfiguration
from azure.ai.ml.exceptions import JobException
from azure.ai.ml.dsl import pipeline
from tests.e2e.utils.constants import (
    COMPONENT_NAME_DATA_JOINER,
    DATA_ASSET_IRIS_PREPROCESSED_MODEL_INPUTS_NO_OVERLAPPING_JOIN_VALUE,
    DATA_ASSET_IRIS_PREPROCESSED_MODEL_INPUTS_WITH_JOIN_COLUMN,
    DATA_ASSET_IRIS_PREPROCESSED_MODEL_OUTPUTS_WITH_JOIN_COLUMN,
    DATA_ASSET_MODEL_INPUTS_JOIN_COLUMN_NAME,
    DATA_ASSET_MODEL_OUTPUTS_JOIN_COLUMN_NAME
)


def _submit_data_joiner_job(
    submit_pipeline_job,
    ml_client: MLClient,
    get_component,
    experiment_name,
    left_input_data,
    left_join_column,
    right_input_data,
    right_join_column,
    expect_failure: bool = False
):
    data_joiner_component = get_component(COMPONENT_NAME_DATA_JOINER)

    @pipeline()
    def _data_joiner_e2e():
        data_joiner_output: Spark = data_joiner_component(
            left_input_data=Input(path=left_input_data, mode='direct', type='mltable'),
            left_join_column=left_join_column,
            right_input_data=Input(path=right_input_data, mode='direct', type='mltable'),
            right_join_column=right_join_column
        )

        data_joiner_output.identity = AmlTokenConfiguration()
        data_joiner_output.resources = {
            'instance_type': 'Standard_E8S_V3',
            'runtime_version': '3.4',
        }

        return {
            'joined_data': data_joiner_output.outputs.joined_data
        }

    pipeline_job = _data_joiner_e2e()
    pipeline_job.outputs.joined_data = Output(
        type='mltable', mode='direct'
    )

    pipeline_job = submit_pipeline_job(
        pipeline_job, experiment_name, expect_failure
    )

    # Wait until the job completes
    try:
        ml_client.jobs.stream(pipeline_job.name)
    except JobException:
        # ignore JobException to return job final status
        pass

    return ml_client.jobs.get(pipeline_job.name)


@pytest.mark.e2e
class TestDataJoinerE2E:
    """Test class."""

    def test_data_joiner_successful(
        self, ml_client: MLClient, get_component, submit_pipeline_job, test_suite_name
    ):
        """Test the happy path scenario for data joiner."""
        pipeline_job = _submit_data_joiner_job(
            submit_pipeline_job,
            ml_client,
            get_component,
            test_suite_name,
            DATA_ASSET_IRIS_PREPROCESSED_MODEL_INPUTS_WITH_JOIN_COLUMN,
            DATA_ASSET_MODEL_INPUTS_JOIN_COLUMN_NAME,
            DATA_ASSET_IRIS_PREPROCESSED_MODEL_OUTPUTS_WITH_JOIN_COLUMN,
            DATA_ASSET_MODEL_OUTPUTS_JOIN_COLUMN_NAME
        )

        assert pipeline_job.status == "Completed"

    def test_data_joiner_empty_result_failed(
        self, ml_client: MLClient, get_component, submit_pipeline_job, test_suite_name
    ):
        """Test data joiner that produces empty result."""
        pipeline_job = _submit_data_joiner_job(
            submit_pipeline_job,
            ml_client,
            get_component,
            test_suite_name,
            DATA_ASSET_IRIS_PREPROCESSED_MODEL_INPUTS_NO_OVERLAPPING_JOIN_VALUE,
            DATA_ASSET_MODEL_INPUTS_JOIN_COLUMN_NAME,
            DATA_ASSET_IRIS_PREPROCESSED_MODEL_OUTPUTS_WITH_JOIN_COLUMN,
            DATA_ASSET_MODEL_OUTPUTS_JOIN_COLUMN_NAME,
            expect_failure=True
        )

        assert pipeline_job.status == "Failed"

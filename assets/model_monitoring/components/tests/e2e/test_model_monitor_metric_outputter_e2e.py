# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains e2e tests for the model data collector preprocessor component."""

import pytest
from azure.ai.ml import Input, MLClient, Output
from azure.ai.ml.entities import Spark, AmlTokenConfiguration
from azure.ai.ml.dsl import pipeline
from tests.e2e.utils.constants import (
    COMPONENT_NAME_METRIC_OUTPUTTER,
    DATA_ASSET_MLTABLE_DATA_DRIFT_SIGNAL_OUTPUT
)

testdata = [('True'), ('False')]


def _submit_mdc_preprocessor_job(
    ml_client: MLClient,
    get_component,
    experiment_name,
    input_data
):
    metric_outputter_component = get_component(COMPONENT_NAME_METRIC_OUTPUTTER)

    @pipeline
    def _metric_outputter_e2e():

        metric_outputter_output: Spark = metric_outputter_component(
            signal_metrics=Input(path=input_data, mode="direct", type="uri_folder"),
            monitor_name="model_outputter2_monitor_name",
            signal_name="model_outputter_signal_name",
            signal_type="my_signal_type",
            metric_timestamp="2023-02-02T00:00:00Z"
        )

        metric_outputter_output.identity = AmlTokenConfiguration()
        metric_outputter_output.resources = {
            'instance_type': 'Standard_E8S_V3',
            'runtime_version': '3.3',
        }

        return {
            'signal_output': metric_outputter_output.outputs.signal_output
        }

    pipeline_job = _metric_outputter_e2e()
    pipeline_job.outputs.signal_output = Output(
        type='uri_folder', mode='direct'
    )

    pipeline_job = ml_client.jobs.create_or_update(
        pipeline_job, experiment_name=experiment_name
    )

    # Wait until the job completes
    ml_client.jobs.stream(pipeline_job.name)
    return ml_client.jobs.get(pipeline_job.name)


@pytest.mark.e2e2
class TestModelMonitorMetricOutputterE2E:
    """Test class."""

    def test_mdc_preprocessor_successful(
        self, ml_client: MLClient, get_component, test_suite_name
    ):
        """Test the happy path scenario for MDC preprocessor."""
        pipeline_job = _submit_mdc_preprocessor_job(
            ml_client,
            get_component,
            test_suite_name,
            DATA_ASSET_MLTABLE_DATA_DRIFT_SIGNAL_OUTPUT
        )

        assert pipeline_job.status == 'Completed'

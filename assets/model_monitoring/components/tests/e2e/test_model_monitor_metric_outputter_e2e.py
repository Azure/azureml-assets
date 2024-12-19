# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains e2e tests for the model data collector preprocessor component."""

import pytest
from azure.ai.ml import Input, MLClient, Output
from azure.ai.ml.entities import Spark, AmlTokenConfiguration
from azure.ai.ml.exceptions import JobException
from azure.ai.ml.dsl import pipeline
from tests.e2e.utils.constants import (
    COMPONENT_NAME_METRIC_OUTPUTTER,
    DATA_ASSET_MLTABLE_DATA_DRIFT_SIGNAL_OUTPUT,
    DATA_ASSET_MLTABLE_SAMPLES_INDEX_OUTPUT
)


def _submit_metric_outputter_job(
    submit_pipeline_job, ml_client: MLClient, get_component, experiment_name, signal_metrics_input,
    samples_index_input, expect_failure: bool = False
):
    metric_outputter_component = get_component(COMPONENT_NAME_METRIC_OUTPUTTER)

    @pipeline
    def _metric_outputter_e2e():

        metric_outputter_output: Spark = metric_outputter_component(
            signal_metrics=Input(path=signal_metrics_input, mode="direct", type="uri_folder"),
            samples_index=Input(path=samples_index_input, mode="direct", type="uri_folder"),
            monitor_name="metric_outputter_monitor_name",
            signal_name="metric_outputter_signal_name",
            signal_type="my_signal_type",
            metric_timestamp="2023-02-02T00:00:00Z",
        )

        metric_outputter_output.identity = AmlTokenConfiguration()
        metric_outputter_output.resources = {
            "instance_type": "Standard_E8S_V3",
            "runtime_version": "3.4",
        }

        return {"signal_output": metric_outputter_output.outputs.signal_output}

    pipeline_job = _metric_outputter_e2e()
    pipeline_job.outputs.signal_output = Output(type="uri_folder", mode="direct")

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
class TestModelMonitorMetricOutputterE2E:
    """Test class."""

    def test_mdc_preprocessor_successful(
        self, ml_client: MLClient, get_component, submit_pipeline_job, test_suite_name
    ):
        """Test the happy path scenario for MDC preprocessor."""
        pipeline_job = _submit_metric_outputter_job(
            submit_pipeline_job,
            ml_client,
            get_component,
            test_suite_name,
            DATA_ASSET_MLTABLE_DATA_DRIFT_SIGNAL_OUTPUT,
            DATA_ASSET_MLTABLE_SAMPLES_INDEX_OUTPUT,
        )

        assert pipeline_job.status == "Completed"

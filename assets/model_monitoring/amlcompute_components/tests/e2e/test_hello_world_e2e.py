# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains e2e tests for the hello world component."""

import pytest
from azure.ai.ml import MLClient, Output
from azure.ai.ml.dsl import pipeline


def _submit_hello_world_model_monitor_job(
    ml_client, get_component, experiment_name
):
    hw_model_monitor = get_component("model_monitor_hello_world")

    @pipeline(
        default_compute="yetatest2"
    )
    def _hello_world_e2e():
        dd_signal_monitor_output = hw_model_monitor()
        return {} # {"signal_output": dd_signal_monitor_output.outputs.signal_output}

    pipeline_job = _hello_world_e2e()
    # pipeline_job.outputs.signal_output = Output(type="mltable", mode="direct")

    pipeline_job = ml_client.jobs.create_or_update(
        pipeline_job, experiment_name=experiment_name
    )

    # Wait until the job completes
    ml_client.jobs.stream(pipeline_job.name)

    return ml_client.jobs.get(pipeline_job.name)


@pytest.mark.e2e
class TestHelloWorld:
    """Test class."""

    def test_hello_world_successful(
        self, ml_client: MLClient, get_component, download_job_output, test_suite_name
    ):
        """Test the happy path scenario where the data has drift and default settings are used."""
        pipeline_job = _submit_hello_world_model_monitor_job(
            ml_client,
            get_component,
            test_suite_name,
        )

        assert pipeline_job.status == "Completed"

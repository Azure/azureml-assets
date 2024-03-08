# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains e2e tests for the data drift model monitor component."""

import pytest
from azure.ai.ml import MLClient, Output
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.exceptions import JobException
from tests.e2e.utils.constants import (
    COMPONENT_NAME_MODEL_TOKEN_STATS_SIGNAL_MONITOR,
    DATA_AGGREGATED_TRACE_DATA
)


def _submit_genai_token_statistics_model_monitor_job(
    ml_client,
    get_component,
    experiment_name,
    aggregated_data
):
    ts_model_monitor = get_component(COMPONENT_NAME_MODEL_TOKEN_STATS_SIGNAL_MONITOR)

    @pipeline()
    def _token_statistics_signal_monitor_e2e():
        token_statistics_signal_monitor_output = ts_model_monitor(
            production_data=aggregated_data,
            signal_name="my_test_create_ts_signal",
            monitor_name="my_test_model_monitor",
            monitor_current_time="2023-04-02T00:00:00Z",
            instance_type="standard_e8s_v3",
        )
        return {
            "signal_output": token_statistics_signal_monitor_output.outputs.signal_output
            }

    pipeline_job = _token_statistics_signal_monitor_e2e()
    pipeline_job.outputs.signal_output = Output(type="uri_folder", mode="direct")
    pipeline_job = ml_client.jobs.create_or_update(
       pipeline_job, experiment_name=experiment_name, skip_validation=True
    )

    # Wait until the job completes
    try:
        ml_client.jobs.stream(pipeline_job.name)
    except JobException:
        # ignore JobException to return job final status
        pass

    return ml_client.jobs.get(pipeline_job.name)


@pytest.mark.e2e
class TestGenAiTSModelMonitor:
    """Test class."""

    def test_monitoring_run_use_defaults_successful(
        self, ml_client: MLClient, get_component, download_job_output,
        test_suite_name
    ):
        """Test the happy path scenario where the data has no drift."""
        pipeline_job = _submit_genai_token_statistics_model_monitor_job(
            ml_client,
            get_component,
            test_suite_name,
            DATA_AGGREGATED_TRACE_DATA,
        )

        assert pipeline_job.status == "Completed"

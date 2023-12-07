# Copyright (c) Microsoft.
# Licensed under the MIT license.

"""This file contains e2e tests for the model performance signal monitor component."""

from azure.ai.ml import MLClient, Output
from azure.ai.ml.dsl import pipeline
from tests.e2e.utils.constants import (
    COMPONENT_NAME_MODEL_PERFORMANCE_SIGNAL_MONITOR,
    DATA_ASSET_IRIS_BASELINE_DATA,
)


def _submit_model_performance_signal_monitor_job(
        ml_client, get_component, test_suite_name, baseline_data, production_data
):
    model_performance_component = get_component(COMPONENT_NAME_MODEL_PERFORMANCE_SIGNAL_MONITOR)

    @pipeline
    def _model_performance_signal_monitor_e2e():
        """Test pipeline definition."""
        model_performance_signal_monitor_output = model_performance_component(
            signal_name="my-model-performance-signal",
            monitor_name="my-model-performance-model-monitor",
            production_data=production_data,
            baseline_data=baseline_data,
            baseline_data_target_column="target",
            production_data_target_column="target",
            task_type="Classification",
            monitor_current_time="2023-01-01T00:00:00Z",
        )

        return {
            "signal_output": model_performance_signal_monitor_output.outputs.signal_output
        }

    pipeline_job = _model_performance_signal_monitor_e2e()
    pipeline_job.outputs.signal_output = Output(type="uri_folder", mode="direct")

    pipeline_job = ml_client.jobs.create_or_update(
        pipeline_job, experiment_name=test_suite_name
    )

    # Wait until the job completes
    ml_client.jobs.stream(pipeline_job.name)

    return ml_client.jobs.get(pipeline_job.name)


class TestModelPerformanceModelMonitor:
    """Test Class"""

    def test_model_performance_signal_monitor_e2e(self, ml_client: MLClient, get_component, test_suite_name):
        """Test the happy path scenario where the data has drift and default settings are used."""
        pipeline_job = _submit_model_performance_signal_monitor_job(
            ml_client,
            get_component,
            test_suite_name,
            DATA_ASSET_IRIS_BASELINE_DATA,
            DATA_ASSET_IRIS_BASELINE_DATA,
        )

        assert pipeline_job.status == "Completed"

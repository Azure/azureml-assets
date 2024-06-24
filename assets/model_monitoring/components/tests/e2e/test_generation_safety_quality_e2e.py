# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains e2e tests for the data drift model monitor component."""

import pytest
from azure.ai.ml import MLClient, Output
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.exceptions import JobException
from tests.e2e.utils.constants import (
    COMPONENT_NAME_GENERATION_SAFETY_QUALITY_SIGNAL_MONITOR,
    DATA_ASSET_GROUNDEDNESS_PREPROCESSED_TARGET_DATA,
    DATA_ASSET_TRACE_LOGS_DATA_WITH_CONTEXT,
)


def _submit_generation_safety_quality_model_monitor_job(
    submit_pipeline_job, ml_client: MLClient, get_component, experiment_name,
    production_data, data_columns_dict: dict = {}, expect_failure: bool = False
):
    generation_safety_quality_signal_monitor = get_component(
        COMPONENT_NAME_GENERATION_SAFETY_QUALITY_SIGNAL_MONITOR
    )

    metrics = [
        "AcceptableCoherenceScorePerInstance",
        "AggregatedCoherencePassRate",
        "AcceptableFluencyScorePerInstance",
        "AggregatedFluencyPassRate",
        "AggregatedGroundednessPassRate",
        "AcceptableGroundednessScorePerInstance",
        "AggregatedRelevancePassRate",
        "AcceptableRelevanceScorePerInstance"
    ]
    metric_names = ",".join(metrics)
    prompt_column_name = data_columns_dict.get('prompt_column_name', None)
    completion_column_name = data_columns_dict.get('completion_column_name', None)
    ground_truth_column_name = data_columns_dict.get('ground_truth_column_name', None)
    context_column_name = data_columns_dict.get('context_column_name', None)

    @pipeline()
    def _generation_safety_quality_signal_monitor_e2e():
        generation_safety_quality_signal_monitor_output = generation_safety_quality_signal_monitor(
            signal_name="my-gsq-signal",
            monitor_name="my-gsq-model-monitor",
            production_data=production_data,
            metric_names=metric_names,
            model_deployment_name="gpt-4",
            sample_rate=1,
            monitor_current_time="2023-02-02T00:00:00Z",
            workspace_connection_arm_id="test_connection",
            prompt_column_name=prompt_column_name,
            completion_column_name=completion_column_name,
            ground_truth_column_name=ground_truth_column_name,
            context_column_name=context_column_name,
        )
        return {
            "signal_output": generation_safety_quality_signal_monitor_output.outputs.signal_output
        }

    pipeline_job = _generation_safety_quality_signal_monitor_e2e()
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


@pytest.mark.gsq_test
@pytest.mark.e2e
class TestGenerationSafetyQualityModelMonitor:
    """Test class for GSQ."""

    def test_generation_safety_quality_successful(
        self, ml_client: MLClient, get_component, submit_pipeline_job, test_suite_name
    ):
        """Test GSQ is successful with traditional expected data."""
        pipeline_job = _submit_generation_safety_quality_model_monitor_job(
            submit_pipeline_job,
            ml_client,
            get_component,
            test_suite_name,
            DATA_ASSET_GROUNDEDNESS_PREPROCESSED_TARGET_DATA,
            {
                "prompt_column_name": "question",
                "completion_column_name": "answer",
            }
        )

        assert pipeline_job.status == "Completed"

    def test_generation_safety_quality_genai_successful(
        self, ml_client: MLClient, get_component, submit_pipeline_job, test_suite_name
    ):
        """Test GSQ is successful with genai trace logs."""
        pipeline_job = _submit_generation_safety_quality_model_monitor_job(
            submit_pipeline_job,
            ml_client,
            get_component,
            test_suite_name,
            DATA_ASSET_TRACE_LOGS_DATA_WITH_CONTEXT,
            {
                "prompt_column_name": "question",
                "completion_column_name": "output",
                "context_column_name": "context",
            }
        )

        assert pipeline_job.status == "Completed"

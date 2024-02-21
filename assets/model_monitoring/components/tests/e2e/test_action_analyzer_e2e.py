# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains e2e tests for the model data collector preprocessor component."""

import pytest
from azure.ai.ml import MLClient, Output
from azure.ai.ml.dsl import pipeline
from tests.e2e.utils.constants import (
    COMPONENT_NAME_ACTION_ANALYZER,
    DATA_ASSET_VIOLATED_METRICS,
    DATA_ASSET_SIGNAL_SCORED_OUTPUT,
    DATA_ASSET_GROUNDEDNESS_PREPROCESSED_TARGET_DATA,
    DATA_ASSET_GENAI_PREPROCESSOR_TRACE_AGGREGATED
)


def _submit_action_analyzer_job(ml_client, get_component, production_data, signal_scored_output):
    action_analyzer_component = get_component(COMPONENT_NAME_ACTION_ANALYZER)

    @pipeline()
    def _action_analyzer_e2e():
        action_analyzer = action_analyzer_component(
            production_data=production_data,
            signal_scored_output=signal_scored_output,
            signal_name="Generation Safety and Quality",
            violated_metrics_names=DATA_ASSET_VIOLATED_METRICS,
            model_deployment_name="gpt-35-turbo-v0301",
            workspace_connection_arm_id="/subscriptions/1aefdc5e-3a7c-4d71-a9f9-f5d3b03be19a/resourceGroups/yuachengtestrg/providers/Microsoft.MachineLearningServices/workspaces/momo-eastus/connections/azureml-rag",
            instance_type="standard_e4s_v3"
        )
        return {"action_output": action_analyzer.outputs.action_output}

    pipeline_job = _action_analyzer_e2e()
    pipeline_job.outputs.action_output = Output(type="uri_folder", mode="direct")

    pipeline_job = ml_client.jobs.create_or_update(
        pipeline_job, experiment_name="action_analyzer_e2e_test", skip_validation=True
    )

    # Wait until the job completes
    ml_client.jobs.stream(pipeline_job.name)

    return ml_client.jobs.get(pipeline_job.name)


@pytest.mark.e2e5
class TestGenerationSafetyModelMonitor:
    """Test class."""

    def test_monitoring_run(
        self, ml_client: MLClient, get_component, download_job_output
    ):
        """Test the happy path scenario."""
        pipeline_job = _submit_action_analyzer_job(
            ml_client,
            get_component,
            DATA_ASSET_GENAI_PREPROCESSOR_TRACE_AGGREGATED,
            DATA_ASSET_SIGNAL_SCORED_OUTPUT
        )

        assert pipeline_job.status == "Completed"
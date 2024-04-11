# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains e2e tests for the model data collector preprocessor component."""

import pytest
from azure.ai.ml import MLClient, Output
from azure.ai.ml.dsl import pipeline
from tests.e2e.utils.constants import (
    COMPONENT_NAME_ACTION_ANALYZER,
    DATA_ASSET_EMPTY
)


def _submit_action_analyzer_job(ml_client, get_component, signal_scored_data, signal_output):
    action_analyzer_component = get_component(COMPONENT_NAME_ACTION_ANALYZER)

    @pipeline()
    def _action_analyzer_e2e():
        action_analyzer = action_analyzer_component(
            signal_scored_data=signal_scored_data,
            signal_name="gsq-signal",
            signal_output=signal_output,
            model_deployment_name="gpt-4-32k",
            workspace_connection_arm_id="test-connection",
            instance_type="standard_e4s_v3",
            aml_deployment_id="test-deployment-id",
            llm_summary_enabled="true"
        )
        return {"action_output": action_analyzer.outputs.action_output}

    pipeline_job = _action_analyzer_e2e()
    pipeline_job.outputs.action_output = Output(type="uri_folder", mode="direct")

    pipeline_job = ml_client.jobs.create_or_update(
        pipeline_job, experiment_name="action_analyzer_e2e_test_0401", skip_validation=True
    )

    # Wait until the job completes
    ml_client.jobs.stream(pipeline_job.name)

    return ml_client.jobs.get(pipeline_job.name)


@pytest.mark.e2e
class TestGenerationSafetyModelMonitor:
    """Test class."""

    def test_monitoring_run(
        self, ml_client: MLClient, get_component, download_job_output
    ):
        """Action anaylyzer can only be tested locally due to workspace connection. So only add test for empty df."""
        pipeline_job = _submit_action_analyzer_job(
            ml_client,
            get_component,
            DATA_ASSET_EMPTY,
            DATA_ASSET_EMPTY
        )

        assert pipeline_job.status == "Completed"
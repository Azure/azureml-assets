# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains e2e tests for the model data collector preprocessor component."""

import pytest
from azure.ai.ml import MLClient, Output
from azure.ai.ml.entities import AmlTokenConfiguration
from azure.ai.ml.exceptions import JobException
from azure.ai.ml.dsl import pipeline
from tests.e2e.utils.constants import (
    COMPONENT_NAME_ACTION_DETECTOR,
    DATA_ASSET_EMPTY,
    DATA_ASSET_SIGNAL_OUTPUT_GSQ
)


def _submit_action_detector_job(ml_client: MLClient, get_component, signal_scored_data, signal_output):
    action_detector_component = get_component(COMPONENT_NAME_ACTION_DETECTOR)

    @pipeline()
    def _action_detector_e2e():
        action_detector = action_detector_component(
            signal_scored_data=signal_scored_data,
            signal_name="gsq-signal",
            signal_output=signal_output,
            model_deployment_name="gpt-4-32k",
            workspace_connection_arm_id="test-connection",
            aml_deployment_id="test-aml-deployment-id",
            query_intention_enabled="true"
        )
        action_detector.identity = AmlTokenConfiguration()
        action_detector.resources = {
            'instance_type': 'Standard_E4S_V3',
            'runtime_version': '3.4',
        }
        return {"action_output": action_detector.outputs.action_output}

    pipeline_job = _action_detector_e2e()
    pipeline_job.outputs.action_output = Output(type="uri_folder", mode="direct")

    pipeline_job = ml_client.jobs.create_or_update(
        pipeline_job, experiment_name="action_detector_e2e_test", skip_validation=True
    )

    # Wait until the job completes
    try:
        ml_client.jobs.stream(pipeline_job.name)
    except JobException:
        # ignore JobException to return job final status
        pass

    return ml_client.jobs.get(pipeline_job.name)


@pytest.mark.e2e
class TestGenerationSafetyModelMonitor:
    """Test class."""

    def test_monitoring_run(
        self, ml_client: MLClient, get_component, download_job_output
    ):
        """Action detector can only be tested locally using workspace connection, so only add test with empty df."""
        pipeline_job = _submit_action_detector_job(
            ml_client,
            get_component,
            DATA_ASSET_EMPTY,
            DATA_ASSET_SIGNAL_OUTPUT_GSQ
        )

        assert pipeline_job.status == "Completed"

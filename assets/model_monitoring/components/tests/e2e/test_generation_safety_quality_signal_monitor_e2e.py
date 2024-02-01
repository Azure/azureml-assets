# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains e2e tests for the model data collector preprocessor component."""

import pytest
from azure.ai.ml import MLClient, Output
from azure.ai.ml.dsl import pipeline
from tests.e2e.utils.constants import (
    COMPONENT_NAME_GENERATION_SAFETY_QUALITY_SIGNAL_MONITOR,
    DATA_ASSET_GROUNDEDNESS_PREPROCESSED_TARGET_DATA
)


def _submit_generation_safety_signal_monitor_job(ml_client,     get_component, target_data):
    generation_safety_model_monitor = get_component(COMPONENT_NAME_GENERATION_SAFETY_QUALITY_SIGNAL_MONITOR)

    @pipeline()
    def _gen_safety_model_monitor_e2e():
        dd_model_monitor_output = generation_safety_model_monitor(
            production_data=target_data,
            signal_name="my_test_generation_safety_signal",
            metric_names="AggregatedGroundednessPassRate, AcceptableCoherenceScorePerInstance, AggregatedCoherencePassRate, AcceptableFluencyScorePerInstance, AggregatedFluencyPassRate, AcceptableSimilarityScorePerInstance, AggregatedSimilarityPassRate, AcceptableRelevanceScorePerInstance, AggregatedRelevancePassRate",  # noqa: E501
            model_deployment_name="gpt-35-turbo-16k",
            sample_rate=1,
            monitor_current_time="2023-02-02T00:00:00Z",
            monitor_name="my_test_model_monitor",
            prompt_column_name="question",
            completion_column_name="answer",
            # we need to have a ws connection in the e2e test environment, this ws has the connection that can run the
            # gsq e2e test: https://ml.azure.com/?tid=72f988bf-86f1-41af-91ab-2d7cd011db47&wsid=/subscriptions/e0fd569c-e34a-4249-8c24-e8d723c7f054/resourceGroups/hawestra-rg/providers/Microsoft.MachineLearningServices/workspaces/hawestra-ws  # noqa: E501
            # workspace_connection_arm_id="/subscriptions/e0fd569c-e34a-4249-8c24-e8d723c7f054/resourceGroups/hawestra-rg/providers/Microsoft.MachineLearningServices/workspaces/hawestra-ws/connections/rai_dev_connection"   # noqa: E501
            workspace_connection_arm_id="/subscriptions/1aefdc5e-3a7c-4d71-a9f9-f5d3b03be19a/resourceGroups/richHoboSynapse/providers/Microsoft.MachineLearningServices/workspaces/hawestra-ws/connections/AOAI_gsp_e2e"   # noqa: E501
        )
        return {"signal_output": dd_model_monitor_output.outputs.signal_output}

    pipeline_job = _gen_safety_model_monitor_e2e()
    pipeline_job.outputs.signal_output = Output(type="uri_folder", mode="direct")

    pipeline_job = ml_client.jobs.create_or_update(
        pipeline_job, experiment_name="generation_safety_e2e_test"
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
        pipeline_job = _submit_generation_safety_signal_monitor_job(
            ml_client,
            get_component,
            DATA_ASSET_GROUNDEDNESS_PREPROCESSED_TARGET_DATA
        )

        assert pipeline_job.status == "Completed"

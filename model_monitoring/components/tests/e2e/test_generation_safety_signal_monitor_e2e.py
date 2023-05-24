# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""This file contains e2e tests for the generation safety model monitor component."""

import pytest
from azure.ai.ml import MLClient, Output
from azure.ai.ml.dsl import pipeline
from tests.e2e.utils.constants import (
    COMPONENT_NAME_GENERATION_SAFETY_QUALITY_SIGNAL_MONITOR,
    DATA_ASSET_GROUNDEDNESS_PREPROCESSED_TARGET_DATA
)


def _submit_generation_safety_signal_monitor_job(ml_client, get_component, target_data):
    generation_safety_model_monitor = get_component(COMPONENT_NAME_GENERATION_SAFETY_QUALITY_SIGNAL_MONITOR)

    @pipeline()
    def _gen_safety_model_monitor_e2e():
        dd_model_monitor_output = generation_safety_model_monitor(
            target_data=target_data,
            signal_name="my_test_generation_safety_signal",
            monitor_name="my_test_model_monitor",
            monitor_current_time="2023-02-02T00:00:00Z",
            model_deployment_name="gpt-35-turbo",
            model_type="gpt-35-turbo",
            endpoint_type="azure_openai_api",
            azure_endpoint_domain_name="aoai-raidev.openai.azure.com",
            authorization_type="key_vault_secret",
            authorization_vault_url="https://modelmonitorin1689166493.vault.azure.net/",
            authorization_secret_name="aoai-raidev-api-key",
            azure_openai_api_version="2023-03-15-preview",
            sample_rate=1.0,
            groundedness_threshold=3
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


@pytest.mark.e2e3
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

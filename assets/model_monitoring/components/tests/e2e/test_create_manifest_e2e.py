# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains e2e tests for the create manifest component."""

import pytest
from azure.ai.ml import MLClient, Output, Input
from azure.ai.ml.entities import Spark, AmlTokenConfiguration
from azure.ai.ml.exceptions import JobException
from azure.ai.ml.dsl import pipeline
from tests.e2e.utils.constants import (
    COMPONENT_NAME_DATA_DRIFT_SIGNAL_MONITOR,
    COMPONENT_NAME_CREATE_MANIFEST,
    COMPONENT_NAME_MDC_PREPROCESSOR,
    DATA_ASSET_IRIS_BASELINE_DATA,
    DATA_ASSET_IRIS_MODEL_INPUTS_WITH_DRIFT,
)


def _submit_data_drift_and_create_manifest_job(
    submit_pipeline_job, ml_client: MLClient, get_component, experiment_name, baseline_data, target_data,
    expect_failure: bool = False
):
    dd_model_monitor = get_component(COMPONENT_NAME_DATA_DRIFT_SIGNAL_MONITOR)
    create_manifest = get_component(COMPONENT_NAME_CREATE_MANIFEST)
    mdc_preprocessor = get_component(COMPONENT_NAME_MDC_PREPROCESSOR)

    @pipeline()
    def _create_manifest_e2e():

        mdc_preprocessor_output = mdc_preprocessor(
            data_window_start="2023-01-29T00:00:00Z",
            data_window_end="2023-02-03T00:00:00Z",
            input_data=Input(path=target_data, mode="direct", type="uri_folder"),
        )

        dd_model_monitor_metrics_output = dd_model_monitor(
            target_data=mdc_preprocessor_output.outputs.preprocessed_input_data,
            baseline_data=baseline_data,
            signal_name="my_test_data_drift_signal",
            monitor_name="my_test_model_monitor",
            monitor_current_time="2023-02-02T00:00:00Z",
            filter_type="Subset",
            filter_value="sepal_length,sepal_width",
            notification_emails="a0142e223_7301@ac5ea71d52378.com",
        )

        create_manifest_output: Spark = create_manifest(
            signal_outputs_1=dd_model_monitor_metrics_output.outputs.signal_output
        )

        mdc_preprocessor_output.identity = AmlTokenConfiguration()
        mdc_preprocessor_output.resources = {
            "instance_type": "Standard_E8S_V3",
            "runtime_version": "3.4",
        }

        create_manifest_output.identity = AmlTokenConfiguration()
        create_manifest_output.resources = {
            "instance_type": "Standard_E8S_V3",
            "runtime_version": "3.4",
        }

        return {
            "model_monitor_metrics_output": create_manifest_output.outputs.model_monitor_metrics_output
        }

    pipeline_job = _create_manifest_e2e()
    pipeline_job.outputs.model_monitor_metrics_output = Output(
        type="uri_folder", mode="direct"
    )

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
class TestCreateManifestE2E:
    """Test class."""

    def test_monitoring_run_use_defaults_data_has_no_drift_successful(
        self, ml_client: MLClient, get_component, download_job_output, submit_pipeline_job, test_suite_name
    ):
        """Test the happy path scenario where the data has drift and default settings are used."""
        pipeline_job = _submit_data_drift_and_create_manifest_job(
            submit_pipeline_job,
            ml_client,
            get_component,
            test_suite_name,
            DATA_ASSET_IRIS_BASELINE_DATA,
            DATA_ASSET_IRIS_MODEL_INPUTS_WITH_DRIFT,
        )

        assert pipeline_job.status == "Completed"

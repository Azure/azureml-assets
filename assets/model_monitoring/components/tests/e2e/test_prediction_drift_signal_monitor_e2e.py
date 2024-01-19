# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains e2e tests for the data drift model monitor component."""

import pytest
from azure.ai.ml import MLClient, Output
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.exceptions import JobException
from tests.e2e.utils.constants import (
    COMPONENT_NAME_PREDICTION_DRIFT_SIGNAL_MONITOR,
    DATA_ASSET_IRIS_BASELINE_DATA,
    DATA_ASSET_IRIS_PREPROCESSED_MODEL_OUTPUTS_NO_DRIFT,
    DATA_ASSET_EMPTY,
    DATA_ASSET_IRIS_PREPROCESSED_MODEL_INPUTS_NO_COMMON_COLUMNS,
    DATA_ASSET_IRIS_BASELINE_INT_DATA_TYPE
)


def _submit_prediction_drift_model_monitor_job(
    ml_client, get_component, experiment_name, baseline_data, target_data
):
    prediction_drift_signal_monitor = get_component(
        COMPONENT_NAME_PREDICTION_DRIFT_SIGNAL_MONITOR
    )

    @pipeline()
    def _prediction_drift_signal_monitor_e2e():
        prediction_drift_signal_monitor_output = prediction_drift_signal_monitor(
            target_data=target_data,
            baseline_data=baseline_data,
            signal_name="my_test_prediction_drift_signal",
            monitor_name="my_test_model_monitor",
            monitor_current_time="2023-02-02T00:00:00Z",
            numerical_threshold=0.5,
            categorical_threshold=0.5,
        )
        return {
            "signal_output": prediction_drift_signal_monitor_output.outputs.signal_output
        }

    pipeline_job = _prediction_drift_signal_monitor_e2e()
    pipeline_job.outputs.signal_output = Output(type="uri_folder", mode="direct")

    pipeline_job = ml_client.jobs.create_or_update(
        pipeline_job, experiment_name=experiment_name
    )

    # Wait until the job completes
    try:
        ml_client.jobs.stream(pipeline_job.name)
    except JobException:
        # ignore JobException to return job final status
        pass

    return ml_client.jobs.get(pipeline_job.name)


@pytest.mark.e2e
class TestPredictionDriftModelMonitor:
    """Test class."""

    def test_monitoring_run_use_defaults_data_has_no_drift_successful(
        self, ml_client: MLClient, get_component, download_job_output, test_suite_name
    ):
        """Test the happy path scenario where the data has drift and default settings are used."""
        pipeline_job = _submit_prediction_drift_model_monitor_job(
            ml_client,
            get_component,
            test_suite_name,
            DATA_ASSET_IRIS_BASELINE_DATA,
            DATA_ASSET_IRIS_PREPROCESSED_MODEL_OUTPUTS_NO_DRIFT,
        )

        assert pipeline_job.status == "Completed"

    def test_monitoring_run_empty_production_data_failed(
        self, ml_client: MLClient, get_component, test_suite_name
    ):
        """Test the scenario where the production data is empty."""
        pipeline_job = _submit_prediction_drift_model_monitor_job(
            ml_client,
            get_component,
            test_suite_name,
            DATA_ASSET_IRIS_BASELINE_DATA,
            DATA_ASSET_EMPTY,
        )

        # empty target data should fail the pipeline
        assert pipeline_job.status == "Failed"

    def test_monitoring_run_no_common_features_production_data_failed(
        self, ml_client: MLClient, get_component, download_job_output,
        test_suite_name
    ):
        """Test the scenario where the production data has no common features with baseline."""
        pipeline_job = _submit_prediction_drift_model_monitor_job(
            ml_client,
            get_component,
            test_suite_name,
            DATA_ASSET_IRIS_BASELINE_DATA,
            DATA_ASSET_IRIS_PREPROCESSED_MODEL_INPUTS_NO_COMMON_COLUMNS,
        )

        # No common columns should fail the job in the feature selector step.
        assert pipeline_job.status == "Failed"

    def test_monitoring_run_empty_production_and_baseline_data(
        self, ml_client: MLClient, get_component, download_job_output,
        test_suite_name
    ):
        """Test the scenario where both input data are empty."""
        pipeline_job = _submit_prediction_drift_model_monitor_job(
            ml_client,
            get_component,
            test_suite_name,
            DATA_ASSET_EMPTY,
            DATA_ASSET_EMPTY,
        )

        # empty production and target data should fail the job
        assert pipeline_job.status == "Failed"

    def test_monitoring_run_mismatched_data_types(self, ml_client: MLClient, get_component, download_job_output,
        test_suite_name
    ):
        """Test the scenario where the data types of the columns in input data are not the same (int vs double for column sepat_length for example)."""
        pipeline_job = _submit_prediction_drift_model_monitor_job(
            ml_client,
            get_component,
            test_suite_name,
            DATA_ASSET_IRIS_BASELINE_INT_DATA_TYPE,
            DATA_ASSET_IRIS_PREPROCESSED_MODEL_OUTPUTS_NO_DRIFT,
        )

        # mismatched data types on columns should fail with no common columns
        assert pipeline_job.status == "Completed"

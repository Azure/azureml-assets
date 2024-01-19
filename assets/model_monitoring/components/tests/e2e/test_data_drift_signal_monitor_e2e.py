# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains e2e tests for the data drift model monitor component."""

import pytest
from azure.ai.ml import MLClient, Output
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.exceptions import JobException
from tests.e2e.utils.constants import (
    COMPONENT_NAME_DATA_DRIFT_SIGNAL_MONITOR,
    DATA_ASSET_IRIS_BASELINE_DATA,
    DATA_ASSET_IRIS_PREPROCESSED_MODEL_INPUTS_NO_DRIFT,
    DATA_ASSET_EMPTY,
    DATA_ASSET_IRIS_BASELINE_INT_DATA_TYPE,
    DATA_ASSET_IRIS_PREPROCESSED_MODEL_INPUTS_NO_DRIFT_INT_DATA,
    DATA_ASSET_IRIS_PREPROCESSED_MODEL_INPUTS_NO_COMMON_COLUMNS,
    DATA_ASSET_IRIS_BASELINE_INT_SINGLE_VALUE_HISTOGRAM,
    DATA_ASSET_IRIS_PREPROCESSED_MODEL_INPUTS_INT_SINGLE_VALUE_HISTOGRAM,
    DATA_ASSET_IRIS_BASELINE_DATA_TYPE_OVERRIDE,
    DATA_ASSET_IRIS_PREPROCESSED_MODEL_INPUTS_TYPE_OVERRIDE,
    DATA_ASSET_WITH_TIMESTAMP_BASELINE_DATA,
    DATA_ASSET_WITH_TIMESTAMP_PRODUCTION_DATA
)


def _submit_data_drift_model_monitor_job(
    ml_client,
    get_component,
    experiment_name,
    baseline_data,
    target_data,
    target_column="target",
    override_numerical_features=None,
    override_categorical_features=None
):
    dd_model_monitor = get_component(COMPONENT_NAME_DATA_DRIFT_SIGNAL_MONITOR)

    @pipeline()
    def _data_drift_signal_monitor_e2e():
        dd_signal_monitor_output = dd_model_monitor(
            target_data=target_data,
            baseline_data=baseline_data,
            signal_name="my_test_create_manifest_signal",
            monitor_name="my_test_model_monitor",
            monitor_current_time="2023-02-02T00:00:00Z",
            target_column=target_column,
            filter_type="TopNByAttribution",
            filter_value="3",
            numerical_threshold=0.5,
            categorical_threshold=0.5,
            override_numerical_features=override_numerical_features,
            override_categorical_features=override_categorical_features
        )
        return {"signal_output": dd_signal_monitor_output.outputs
                .signal_output}

    pipeline_job = _data_drift_signal_monitor_e2e()
    pipeline_job.outputs.signal_output = Output(type="uri_folder",
                                                mode="direct")

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
class TestDataDriftModelMonitor:
    """Test class."""

    def test_monitoring_run_use_defaults_data_has_no_drift_successful(
        self, ml_client: MLClient, get_component, download_job_output,
        test_suite_name
    ):
        """Test the happy path scenario where the data has no drift."""
        pipeline_job = _submit_data_drift_model_monitor_job(
            ml_client,
            get_component,
            test_suite_name,
            DATA_ASSET_IRIS_BASELINE_DATA,
            DATA_ASSET_IRIS_PREPROCESSED_MODEL_INPUTS_NO_DRIFT,
        )

        assert pipeline_job.status == "Completed"

    def test_monitoring_run_empty_production_data_failed(
        self, ml_client: MLClient, get_component, download_job_output,
        test_suite_name
    ):
        """Test the scenario where the production data is empty."""
        pipeline_job = _submit_data_drift_model_monitor_job(
            ml_client,
            get_component,
            test_suite_name,
            DATA_ASSET_IRIS_BASELINE_DATA,
            DATA_ASSET_EMPTY,
        )

        # empty production data should fail the job
        assert pipeline_job.status == "Failed"

    def test_monitoring_run_no_common_features_production_data_failed(
        self, ml_client: MLClient, get_component, download_job_output,
        test_suite_name
    ):
        """Test the scenario where the production data has no common features with baseline."""
        pipeline_job = _submit_data_drift_model_monitor_job(
            ml_client,
            get_component,
            test_suite_name,
            DATA_ASSET_IRIS_BASELINE_DATA,
            DATA_ASSET_IRIS_PREPROCESSED_MODEL_INPUTS_NO_COMMON_COLUMNS,
        )

        # No common columns should fail the job in the feature selector step.
        assert pipeline_job.status == "Failed"

    def test_monitoring_run_use_int_data_has_no_drift_successful(
        self, ml_client: MLClient, get_component, download_job_output,
        test_suite_name
    ):
        """Test the happy path scenario with int data."""
        pipeline_job = _submit_data_drift_model_monitor_job(
            ml_client,
            get_component,
            test_suite_name,
            DATA_ASSET_IRIS_BASELINE_INT_DATA_TYPE,
            DATA_ASSET_IRIS_PREPROCESSED_MODEL_INPUTS_NO_DRIFT_INT_DATA
        )

        assert pipeline_job.status == "Completed"

    def test_monitoring_run_empty_production_and_baseline_data(
        self, ml_client: MLClient, get_component, download_job_output,
        test_suite_name
    ):
        """Test the scenario where the production data is empty."""
        pipeline_job = _submit_data_drift_model_monitor_job(
            ml_client,
            get_component,
            test_suite_name,
            DATA_ASSET_EMPTY,
            DATA_ASSET_EMPTY,
        )

        # empty production and target data should fail the job
        assert pipeline_job.status == "Failed"

    def test_monitoring_run_int_single_distinct_value_histogram(
        self, ml_client: MLClient, get_component, download_job_output,
        test_suite_name
    ):
        """Test the scenario where the production data has a column with only one distinct value."""
        pipeline_job = _submit_data_drift_model_monitor_job(
            ml_client,
            get_component,
            test_suite_name,
            DATA_ASSET_IRIS_BASELINE_INT_SINGLE_VALUE_HISTOGRAM,
            DATA_ASSET_IRIS_PREPROCESSED_MODEL_INPUTS_INT_SINGLE_VALUE_HISTOGRAM,
        )

        assert pipeline_job.status == "Completed"

    def test_monitoring_run_successful_with_datatype_override(
        self, ml_client: MLClient, get_component, download_job_output, test_suite_name
    ):
        """Test the happy path scenario with datatype override."""
        pipeline_job = _submit_data_drift_model_monitor_job(
            ml_client,
            get_component,
            test_suite_name,
            DATA_ASSET_IRIS_BASELINE_DATA_TYPE_OVERRIDE,
            DATA_ASSET_IRIS_PREPROCESSED_MODEL_INPUTS_TYPE_OVERRIDE,
            "target",
            "sepal_width",
            "petal_length"
        )

        assert pipeline_job.status == "Completed"

    def test_monitoring_run_successful_with_datatype_override(
        self, ml_client: MLClient, get_component, download_job_output, test_suite_name
    ):
        """Test the happy path scenario with datatype override."""
        pipeline_job = _submit_data_drift_model_monitor_job(
            ml_client,
            get_component,
            test_suite_name,
            DATA_ASSET_WITH_TIMESTAMP_BASELINE_DATA,
            DATA_ASSET_WITH_TIMESTAMP_PRODUCTION_DATA,
            "DEFAULT_NEXT_MONTH"
        )

        assert pipeline_job.status == "Completed"

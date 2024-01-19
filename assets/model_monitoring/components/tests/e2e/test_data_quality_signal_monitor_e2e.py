# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains e2e tests for the data quality signal monitor component."""

import pytest
from azure.ai.ml import MLClient, Output
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.exceptions import JobException
from tests.e2e.utils.constants import (
    COMPONENT_NAME_DATA_QUALITY_SIGNAL_MONITOR,
    DATA_ASSET_EMPTY,
    DATA_ASSET_IRIS_BASELINE_DATA,
    DATA_ASSET_IRIS_PREPROCESSED_MODEL_INPUTS_NO_DRIFT,
    DATA_ASSET_VALID_DATATYPE,
    DATA_ASSET_IRIS_BASELINE_DATA_TYPE_OVERRIDE,
    DATA_ASSET_IRIS_PREPROCESSED_MODEL_INPUTS_TYPE_OVERRIDE,
    DATA_ASSET_WITH_TIMESTAMP_BASELINE_DATA,
    DATA_ASSET_WITH_TIMESTAMP_PRODUCTION_DATA
)


def _submit_data_quality_signal_monitor_job(
    ml_client,
    get_component,
    experiment_name,
    baseline_data,
    target_data,
    target_column=None,
    filter_type=None,
    filter_value=None,
    override_numerical_features=None,
    override_categorical_features=None
):
    dd_signal_monitor = get_component(COMPONENT_NAME_DATA_QUALITY_SIGNAL_MONITOR)

    @pipeline()
    def _data_quality_signal_monitor_e2e():
        dd_signal_monitor_output = dd_signal_monitor(
            target_data=target_data,
            baseline_data=baseline_data,
            signal_name="my_test_data_quality_signal",
            monitor_name="my_test_model_monitor",
            monitor_current_time="2023-02-02T00:00:00Z",
            target_column=target_column,
            filter_type=filter_type,
            filter_value=filter_value,
            numerical_threshold=0.5,
            categorical_threshold=0.5,
            override_numerical_features=override_numerical_features,
            override_categorical_features=override_categorical_features
        )
        return {"signal_output": dd_signal_monitor_output.outputs.signal_output}

    pipeline_job = _data_quality_signal_monitor_e2e()
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
class TestDataQualityModelMonitor:
    """Test class."""

    def test_monitoring_run_use_defaults_data_has_no_drift_successful(
        self, ml_client: MLClient, get_component, download_job_output, test_suite_name
    ):
        """Test the happy path scenario where the data has drift and default settings are used."""
        pipeline_job = _submit_data_quality_signal_monitor_job(
            ml_client,
            get_component,
            test_suite_name,
            DATA_ASSET_IRIS_BASELINE_DATA,
            DATA_ASSET_IRIS_PREPROCESSED_MODEL_INPUTS_NO_DRIFT,
            "target",
            "TopNByAttribution",
            "3"
        )

        assert pipeline_job.status == "Completed"

    def test_monitoring_run_successful_with_datatype_override(
        self, ml_client: MLClient, get_component, download_job_output, test_suite_name
    ):
        """Test the happy path scenario with datatype override."""
        pipeline_job = _submit_data_quality_signal_monitor_job(
            ml_client,
            get_component,
            test_suite_name,
            DATA_ASSET_IRIS_BASELINE_DATA_TYPE_OVERRIDE,
            DATA_ASSET_IRIS_PREPROCESSED_MODEL_INPUTS_TYPE_OVERRIDE,
            "target",
            "TopNByAttribution",
            "3",
            "sepal_width",
            "petal_length"
        )

        assert pipeline_job.status == "Completed"

    def test_monitoring_run_successful_with_timestamp_data(
        self, ml_client: MLClient, get_component, download_job_output, test_suite_name
    ):
        """Test the happy path scenario with timestamp data."""
        pipeline_job = _submit_data_quality_signal_monitor_job(
            ml_client,
            get_component,
            test_suite_name,
            DATA_ASSET_WITH_TIMESTAMP_BASELINE_DATA,
            DATA_ASSET_WITH_TIMESTAMP_PRODUCTION_DATA,
            "DEFAULT_NEXT_MONTH",
            "TopNByAttribution",
            "3"
        )

        assert pipeline_job.status == "Completed"

    def test_monitoring_run_use_defaults_empty_production_data_failed(
        self, ml_client: MLClient, get_component, test_suite_name
    ):
        """Test the scenario where the production data is empty."""
        pipeline_job = _submit_data_quality_signal_monitor_job(
            ml_client,
            get_component,
            test_suite_name,
            DATA_ASSET_IRIS_BASELINE_DATA,
            DATA_ASSET_EMPTY,
            "target",
            "TopNByAttribution",
            "3"
        )

        # empty target data should cause the pipeline to fail
        assert pipeline_job.status == "Failed"

    def test_monitoring_run_add_more_valid_datatype_data_successful(
        self, ml_client: MLClient, get_component, test_suite_name
    ):
        """Test the scenario where the datatype contains timestamp and boolean."""
        # The test case does not choose the target_column because of a bug in feature_importance
        # component did not support timestamp type. So we do not select target_column for now for the test
        pipeline_job = _submit_data_quality_signal_monitor_job(
            ml_client,
            get_component,
            test_suite_name,
            DATA_ASSET_VALID_DATATYPE,
            DATA_ASSET_VALID_DATATYPE,
            None,
            "All",
            None
        )

        assert pipeline_job.status == "Completed"

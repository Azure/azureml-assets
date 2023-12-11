# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains e2e tests for the data quality signal monitor component."""

import pytest
from azure.ai.ml import MLClient, Output
from azure.ai.ml.dsl import pipeline
from tests.e2e.utils.constants import (
    COMPONENT_NAME_MODEL_PERFORMANCE_SIGNAL_MONITOR,
    DATA_ASSET_IRIS_BASELINE_DATA,
)


def _submit_model_performance_signal_monitor_job(
    ml_client,
    get_component,
    experiment_name,
    task,
    baseline_data,
    baseline_data_target_column,
    production_data,
    production_data_target_column,
    regression_rmse_threshold=None,
    regression_meanabserror_threshold=None,
    classification_precision_threshold=None,
    classification_accuracy_threshold=None,
    classification_recall_threshold=None,

):
    mp_signal_monitor = get_component(COMPONENT_NAME_MODEL_PERFORMANCE_SIGNAL_MONITOR)

    @pipeline()
    def _model_performance_signal_monitor_e2e():
        mp_signal_monitor_output = mp_signal_monitor(
            task=task,
            baseline_data=baseline_data,
            baseline_data_target_column=baseline_data_target_column,
            production_data=production_data,
            production_data_target_column=production_data_target_column,
            regression_rmse_threshold=regression_rmse_threshold,
            regression_meanabserror_threshold=regression_meanabserror_threshold,
            classification_precision_threshold=classification_precision_threshold,
            classification_accuracy_threshold=classification_accuracy_threshold,
            classification_recall_threshold=classification_recall_threshold,
            signal_name="my_test_model_performance_signal",
            monitor_name="my_test_model_monitor",
            monitor_current_time="2023-02-02T00:00:00Z",
        )
        return {"signal_output": mp_signal_monitor_output.outputs.signal_output}

    pipeline_job = _model_performance_signal_monitor_e2e()
    pipeline_job.outputs.signal_output = Output(type="uri_folder", mode="direct")

    pipeline_job = ml_client.jobs.create_or_update(
        pipeline_job, experiment_name=experiment_name
    )

    # Wait until the job completes
    ml_client.jobs.stream(pipeline_job.name)

    return ml_client.jobs.get(pipeline_job.name)


@pytest.mark.e2e
class TestModelPerformanceModelMonitor:
    """Test class."""

    def test_monitoring_regression_successful(
        self, ml_client: MLClient, get_component, download_job_output, test_suite_name
    ):
        """Test model performance on regression model."""
        pipeline_job = _submit_model_performance_signal_monitor_job(
            ml_client,
            get_component,
            test_suite_name,
            "tabular-regression",
            DATA_ASSET_IRIS_BASELINE_DATA,
            "sepal_length",
            DATA_ASSET_IRIS_BASELINE_DATA,
            "sepal_length",
            0.1,
            0.1
        )

        assert pipeline_job.status == "Completed"


    def test_monitoring_classification_successful(
        self, ml_client: MLClient, get_component, download_job_output, test_suite_name
    ):
        """Test model performance on classification model."""
        pipeline_job = _submit_model_performance_signal_monitor_job(
            ml_client,
            get_component,
            test_suite_name,
            "tabular-classification",
            DATA_ASSET_IRIS_BASELINE_DATA,
            "target",
            DATA_ASSET_IRIS_BASELINE_DATA,
            "target",
            None,
            None,
            0.9,
            0.9,
            0.9,
        )

        assert pipeline_job.status == "Completed"       


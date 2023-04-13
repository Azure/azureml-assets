# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests running a sample job in the automl-dnn environment."""
import os
import time
from pathlib import Path
from azure.ai.ml import automl
from azure.ai.ml import MLClient
from azure.ai.ml import Input
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Environment, BuildContext
from azure.identity import AzureCliCredential

BUILD_CONTEXT = Path("../context")
JOB_SOURCE_CODE = "src"
TIMEOUT_MINUTES = os.environ.get("timeout_minutes", 60)

def test_azureml_automl():
    this_dir = Path(__file__).parent

    subscription_id = os.environ.get("subscription_id")
    resource_group = os.environ.get("resource_group")
    workspace_name = os.environ.get("workspace")

    ml_client = MLClient(
        AzureCliCredential(), subscription_id, resource_group, workspace_name
    )

    env_name = "AutoML-Non-Prod-DNN"

    env_docker_context = Environment(
        build=BuildContext(path=this_dir / BUILD_CONTEXT),
        name=env_name,
        description="AutoML environment created from a Docker context.",
    )
    ml_client.environments.create_or_update(env_docker_context)

    my_training_data_input = Input(
        type=AssetTypes.MLTABLE, path=f"{this_dir}/training-mltable-folder"
    )

    forecasting_job = automl.text_classification(
        compute=os.environ.get("cpu_cluster"),
        experiment_name="AutoMLDNNCPUExperiment",
        training_data=my_training_data_input,
        target_column_name="count",
        primary_metric="accuracy",
        n_cross_validations=10,
        properties={'_automl_internal_scenario': 'non-prod', 'enable_code_generation': True},
    )

    # Limits are all optional
    forecasting_job.set_limits(
        timeout_minutes=60,
        trial_timeout_minutes=30,
        max_trials=4,
        max_concurrent_trials=4,
    )

    # Specialized properties for Time Series Forecasting training
    forecasting_job.set_forecast_settings(
        time_column_name="date", forecast_horizon=14, frequency="D"
    )

    # Enable Dnn training and allow only TCNForecaster model
    forecasting_job.set_training(
        allowed_training_algorithms=["TCNForecaster"], enable_dnn_training=True
    )

    returned_job = ml_client.create_or_update(forecasting_job)
    assert returned_job is not None

    # Poll until final status is reached, or timed out
    timeout = time.time() + (TIMEOUT_MINUTES * 60)
    while time.time() <= timeout:
        current_status = ml_client.jobs.get(returned_job.name).status
        if current_status in ["Completed", "Failed"]:
            break
        time.sleep(30)  # sleep 30 seconds

    assert current_status == "Completed"
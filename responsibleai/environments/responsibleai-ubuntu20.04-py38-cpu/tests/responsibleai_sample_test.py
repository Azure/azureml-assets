# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests running a sample job in the responsibleai 0.22 environment."""
import mlflow
import os
import time
from pathlib import Path
from azure.ai.ml import MLClient
from azure.ai.ml import command, Input
from azure.ai.ml._restclient.models import JobStatus
from azure.ai.ml.entities import Environment, BuildContext
from azure.ai.ml import automl
from azure.ai.ml.constants import AssetTypes
from azure.identity import AzureCliCredential

BUILD_CONTEXT = Path("../context")
JOB_SOURCE_CODE = "src"
DATA_SOURCE = "data/training-mltable-folder"
TIMEOUT_MINUTES = os.environ.get("timeout_minutes", 30)
STD_LOG = Path("artifacts/user_logs/std_log.txt")


def verify_automl_rai_run(ml_client, automl_job):

    # Obtain the tracking URL from MLClient
    MLFLOW_TRACKING_URI = ml_client.workspaces.get(
        name=ml_client.workspace_name
    ).mlflow_tracking_uri

    print(MLFLOW_TRACKING_URI)

    # Set the MLFLOW TRACKING URI
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    print("\nCurrent tracking uri: {}".format(mlflow.get_tracking_uri()))

    from mlflow.tracking.client import MlflowClient

    # Initialize MLFlow client
    mlflow_client = MlflowClient()

    mlflow_automl_rai_run = mlflow_client.get_run(automl_job.name + "_RAI")
    # Poll until final status is reached, or timed out
    timeout = time.time() + (TIMEOUT_MINUTES * 60)
    while time.time() <= timeout:
        # assert mlflow_automl_rai_run.info.status == 'FINISHED'
        if mlflow_automl_rai_run.info.status in [
                'FINISHED', 'FAILED', 'KILLED']:
            break
        time.sleep(30)  # sleep 30 seconds

    assert mlflow_automl_rai_run.info.status == 'FINISHED'


def test_responsibleai():
    """Tests a sample job using responsibleai image as the environment."""
    this_dir = Path(__file__).parent

    subscription_id = os.environ.get("subscription_id")
    resource_group = os.environ.get("resource_group")
    workspace_name = os.environ.get("workspace")

    ml_client = MLClient(
        AzureCliCredential(), subscription_id, resource_group, workspace_name
    )

    env_name = "responsibleai"

    env_docker_context = Environment(
        build=BuildContext(path=this_dir / BUILD_CONTEXT),
        name=env_name,
        description="ResponsibleAI environment created from a Docker context.",
    )
    ml_client.environments.create_or_update(env_docker_context)

    # create the command
    job = command(
        code=this_dir / JOB_SOURCE_CODE,  # local path where the code is stored
        command="python main.py --diabetes-csv ${{inputs.diabetes}}",
        inputs={
            "diabetes": Input(
                type="uri_file",
                path="https://azuremlexamples.blob.core.windows.net/datasets/diabetes.csv",
            )
        },
        environment=f"{env_name}@latest",
        compute=os.environ.get("cpu_cluster"),
        display_name="responsibleai-diabetes-example",
        description="A test run of the responsibleai curated environment",
        experiment_name="responsibleaiExperiment"
    )

    returned_job = ml_client.create_or_update(job)
    assert returned_job is not None

    # Poll until final status is reached, or timed out
    timeout = time.time() + (TIMEOUT_MINUTES * 60)
    while time.time() <= timeout:
        current_status = ml_client.jobs.get(returned_job.name).status
        if current_status in [JobStatus.COMPLETED, JobStatus.FAILED]:
            break
        time.sleep(30)  # sleep 30 seconds

    if current_status == JobStatus.FAILED:
        ml_client.jobs.download(returned_job.name)
        if STD_LOG.exists():
            print(f"*** BEGIN {STD_LOG} ***")
            with open(STD_LOG, "r") as f:
                print(f.read(), end="")
            print(f"*** END {STD_LOG} ***")
        else:
            ml_client.jobs.stream(returned_job.name)

    assert current_status == "Completed"


def test_responsibleai_automl_regression():
    """Tests a sample automl job using responsibleai image as the environment."""
    this_dir = Path(__file__).parent
    subscription_id = os.environ.get("subscription_id")
    resource_group = os.environ.get("resource_group")
    workspace_name = os.environ.get("workspace")

    ml_client = MLClient(
        AzureCliCredential(), subscription_id, resource_group, workspace_name
    )
    # general job parameters
    max_trials = 1
    exp_name = "dpv2-regression-experiment"
    # Training MLTable defined locally, with local data to be uploaded
    my_training_data_input = Input(
        type=AssetTypes.MLTABLE, path=(this_dir / DATA_SOURCE)
    )

    # Create the AutoML regression job with the related factory-function.
    regression_job = automl.regression(
        compute=os.environ.get("cpu_cluster"),
        experiment_name=exp_name,
        training_data=my_training_data_input,
        target_column_name="ERP",
        primary_metric="R2Score",
        n_cross_validations=5,
        enable_model_explainability=True,
        tags={"my_custom_tag": "My custom value"},
        properties={"_aml_internal_automl_best_rai": True}
    )

    # Limits are all optional
    regression_job.set_limits(
        timeout_minutes=60,
        trial_timeout_minutes=20,
        max_trials=max_trials,
        enable_early_termination=True,
    )

    # Submit the AutoML job
    returned_job = ml_client.jobs.create_or_update(
        regression_job
    )  # submit the job to the backend

    print(f"Created job: {returned_job}")
    assert returned_job is not None

    # Poll until final status is reached, or timed out
    timeout = time.time() + (TIMEOUT_MINUTES * 60)
    while time.time() <= timeout:
        current_status = ml_client.jobs.get(returned_job.name).status
        if current_status in [JobStatus.COMPLETED, JobStatus.FAILED]:
            break
        time.sleep(30)  # sleep 30 seconds

    if current_status == JobStatus.FAILED:
        ml_client.jobs.download(returned_job.name)
        if STD_LOG.exists():
            print(f"*** BEGIN {STD_LOG} ***")
            with open(STD_LOG, "r") as f:
                print(f.read(), end="")
            print(f"*** END {STD_LOG} ***")
        else:
            ml_client.jobs.stream(returned_job.name)

    assert current_status == "Completed"
    verify_automl_rai_run(ml_client, returned_job)

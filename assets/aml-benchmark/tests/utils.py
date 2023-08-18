# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Utility file for tests."""

import os
from typing import Dict, Any, Optional

from azure.ai.ml import MLClient, load_job
from azure.ai.ml.entities import Job
from azure.identity import DefaultAzureCredential, AzureCliCredential
import mlflow


class Constants:
    """Class to hold all constants."""

    SAMPLER_INPUT_FILE_1 = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data/sampler_data_1.jsonl"
    )
    SAMPLER_INPUT_FILE_2 = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data/sampler_data_2.jsonl"
    )
    OUTPUT_DIR = "{output_dir}/named-outputs/{output_name}"
    OUTPUT_FILE_PATH = OUTPUT_DIR + "/{output_file_name}"
    TEST_REGISTRY_NAME = "Benchmark-Test"
    REMOTE_FILE_URL = "https://raw.githubusercontent.com/Azure/azureml-examples/main/\
v1/python-sdk/tutorials/automl-with-azureml/forecasting-bike-share/bike-no.csv"


def get_current_path() -> str:
    """Get the current path of the script."""
    return os.path.dirname(os.path.abspath(__file__))


def get_mlclient(registry_name: Optional[str] = None) -> MLClient:
    """
    Get the MLClient instance for either workspace or registry.

    If `registry_name` is None, then the MLClient instance will be created for workspace.
    Else, the MLClient instance will be created for registry.

    :param registry_name: Name of the registry.
    :return: MLClient instance.
    """
    try:
        credential = AzureCliCredential()
    except Exception as ex:
        print(f"Unable to authenticate via Azure CLI:\n{ex}")
        credential = DefaultAzureCredential()
        credential.get_token("https://management.azure.com/.default")
    try:
        ml_client = MLClient.from_config(credential=credential)
    except Exception:
        ml_client = MLClient(
            credential=credential,
            subscription_id=os.environ.get("subscription_id"),
            resource_group_name=os.environ.get("resource_group"),
            workspace_name=os.environ.get("workspace"),
        )

    if registry_name:
        ml_client = MLClient(
            credential,
            ml_client.subscription_id,
            ml_client.resource_group_name,
            registry_name=registry_name,
        )
    return ml_client


def load_yaml_pipeline(file_name: str, base_path: str = "pipelines") -> Job:
    """
    Load the YAML pipeline and return the pipeline job.

    :param file_name: name of the file to load.
    :param base_path: name of the base path to load the file from.

    :return: The pipeline job
    """
    component_path = os.path.join(get_current_path(), base_path, file_name)
    pipeline_job = load_job(component_path)
    return pipeline_job


def download_outputs(
    job_name: str,
    download_path: str,
    output_name: str = None,
    all: bool = False,
) -> None:
    """
    Download outputs of the job.

    :param job_name: Name of a job.
    :type job_name: str
    :param download_path: Local path as download destination.
    :type download_path: str, optional
    :param output_name: Named output to download, defaults to None.
    :type output_name: str, optional
    :param all: Whether to download logs and all named outputs, defaults to False.
    :type all: bool, optional
    :return: None
    :rtype: NoneType
    """
    ml_client = get_mlclient()
    ml_client.jobs.download(
        name=job_name,
        output_name=output_name,
        download_path=download_path,
        all=all,
    )


def get_mlflow_logged_params(job_name: str, exp_name: str) -> Dict[str, Any]:
    """
    Get the logged parameters from the mlflow run for an azureml job.

    :param job_name: Name of the azureml job.
    :param exp_name: Name of the azureml experiment.
    :return: Logged parameters.
    """
    ml_client = get_mlclient()

    # get and set the mlflow tracking uri
    mlflow_tracking_uri = ml_client.workspaces.get(
        ml_client.workspace_name
    ).mlflow_tracking_uri
    mlflow.set_tracking_uri(mlflow_tracking_uri)

    # get the logged parameters
    filter = f"tags.mlflow.parentRunId='{job_name}'"
    runs = mlflow.search_runs(
        experiment_names=[exp_name], filter_string=filter, output_format="list"
    )
    logged_params = runs[0].data.params

    return logged_params

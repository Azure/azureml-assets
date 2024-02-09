# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Utility file for tests."""

import os
import subprocess
from typing import Dict, Any, Optional, List
import hashlib
import time
import uuid

from azure.ai.ml import MLClient, load_job
from azure.ai.ml.entities import Job
from azure.identity import DefaultAzureCredential, AzureCliCredential
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Model,
    CodeConfiguration,
)
from azure.ai.ml.entities import WorkspaceConnection
from azure.ai.ml.entities import AccessKeyConfiguration
from azureml._common._error_response._error_response_constants import ErrorCodes
import mlflow


class Constants:
    """Class to hold all constants."""

    SAMPLER_INPUT_FILE_1 = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data/sampler_data_1.jsonl"
    )
    SAMPLER_INPUT_FILE_2 = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data/sampler_data_2.jsonl"
    )
    PERF_INPUT_FILE_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data/perf_metrics_data.jsonl"
    )
    BATCH_INFERENCE_PREPARER_FILE_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data/sample_batch_input.jsonl"
    )
    BATCH_INFERENCE_PREPARER_FILE_PATH_VISION = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data/sample_batch_input_vision.jsonl"
    )
    BATCH_OUTPUT_FORMATTER_FILE_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", "batch_inference_output.jsonl"
    )
    BATCH_INFERENCE_FILE_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", "batch_inference_input.json"
    )
    BATCH_INFERENCE_FILE_PATH_VISION = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", "batch_inference_input_vision.jsonl"
    )
    PROCESS_SAMPLE_EXAMPLES_INPUT_FILE = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data/process_sample_examples.jsonl"
    )
    PREPROCESS_SAMPLE_EXAMPLES_EXPECTED_OUTPUT_FILE = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data/sample_examples_expected_preprocessed_outputs.jsonl"
    )
    CUSTOM_PREPROCESSOR_SCRIPT_PATH = os.path.join(
        os.path.dirname(__file__), '../scripts/custom_dataset_preprocessors'
    )
    POSTPROCESS_SAMPLE_EXAMPLES_INFERENCE_FILE = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data/process_inference_sample_examples.jsonl"
    )
    POSTPROCESS_SAMPLE_EXAMPLES_GROUND_TRUTH_FILE = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data/process_ground_truth_sample_examples.jsonl"
    )
    POSTPROCESS_SAMPLE_EXAMPLES_EXPECTED_OUTPUT_FILE = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data/sample_examples_expected_postprocessed_outputs.jsonl"
    )
    PROMPTCRAFTER_SAMPLE_INPUT_FILE = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data/test_data_prompt_crafter/inferencesample.jsonl"
    )
    PROMPTCRAFTER_SAMPLE_FEWSHOT_FILE = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data/test_data_prompt_crafter/fewshotsample.jsonl"
    )
    REFERENCES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "references")
    OUTPUT_DIR = "{output_dir}/named-outputs/{output_name}"
    OUTPUT_FILE_PATH = OUTPUT_DIR + "/{output_file_name}"
    TEST_REGISTRY_NAME = "Benchmark-Test"
    MATH_DATASET_LOADER_SCRIPT = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "scripts", "data_loaders", "math.py"
    )


def _get_runs(job_name: str, exp_name: str) -> List[mlflow.entities.Run]:
    """
    Get the runs from the mlflow for an azureml job.

    :param job_name: Name of the azureml job.
    :param exp_name: Name of the azureml experiment.
    :return: List of runs.
    """
    ml_client = get_mlclient()

    # get and set the mlflow tracking uri
    mlflow_tracking_uri = ml_client.workspaces.get(
        ml_client.workspace_name
    ).mlflow_tracking_uri
    mlflow.set_tracking_uri(mlflow_tracking_uri)

    # get the runs
    filter = f"tags.mlflow.parentRunId='{job_name}'"
    runs = mlflow.search_runs(
        experiment_names=[exp_name], filter_string=filter, output_format="list"
    )
    return runs


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
        credential.get_token("https://management.azure.com/.default")
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
    runs = _get_runs(job_name, exp_name)
    logged_params = runs[0].data.params
    return logged_params


def get_mlflow_logged_metrics(job_name: str, exp_name: str) -> Dict[str, Any]:
    """
    Get the logged metrics from the mlflow run for an azureml job.

    :param job_name: Name of the azureml job.
    :param exp_name: Name of the azureml experiment.
    :return: Logged metrics.
    """
    runs = _get_runs(job_name, exp_name)
    logged_metrics = runs[0].data.metrics
    return logged_metrics


def get_src_dir() -> str:
    """Get the source directory for component code."""
    cwd = os.getcwd()
    if os.path.basename(cwd) == "aml-benchmark":
        src_dir = "components/src"
    elif os.path.basename(cwd) == "azureml-assets":
        # when running tests locally
        src_dir = "assets/aml-benchmark/components/src"
    else:
        # when running tests from workflow
        src_dir = os.path.join(os.path.dirname(cwd), "src")
    return src_dir


def run_command(command: str) -> None:
    """
    Run the command in the shell.

    :param command: The command to run.
    :return: None
    """
    _ = subprocess.check_output(
        command,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        shell=True,
    )


def assert_exception_mssg(exception_message: str, expected_exception_mssg: str) -> None:
    """
    Assert that the exception message contains the expected exception message and the expected error code.

    :param exception_message: The exception message.
    :param expected_exception_mssg: The expected exception message.
    """
    assert expected_exception_mssg in exception_message
    assert ErrorCodes.USER_ERROR in exception_message
    assert ErrorCodes.SYSTEM_ERROR not in exception_message


def assert_logged_params(job_name: str, exp_name: str, **expected_params: Any) -> None:
    """
    Assert that the logged parameters are as expected.

    If a file path or a list of file paths is provided in the `expected_params` value, the checksum of the
    file(s) is calculated and then asserted. If `None` is provided in the `expected_params`
    value, the parameter is asserted to haven't been logged.

    :param job_name: Name of the azureml job.
    :param exp_name: Name of the azureml experiment.
    :param **expected_params: The expected key-value pairs of parameters.
    :return: None
    """
    logged_params = get_mlflow_logged_params(job_name, exp_name)

    params = {}
    for key, value in expected_params.items():
        if isinstance(value, str) and os.path.isfile(value):
            # calculate checksum of input dataset
            checksum = hashlib.sha256(open(value, "rb").read()).hexdigest()
            params[key] = checksum
        elif isinstance(value, list) and all(isinstance(item, str) and os.path.isfile(item) for item in value):
            # calculate checksum of input dataset
            checksum = hashlib.sha256(b"".join(open(item, "rb").read() for item in value)).hexdigest()
            params[key] = checksum
        else:
            params[key] = value

    for key, value in params.items():
        if value is None:
            assert key not in logged_params
        else:
            assert logged_params[key] == str(value)


def _deploy_endpoint(ml_client, endpoint_name):
    endpoint = ManagedOnlineEndpoint(
        name=endpoint_name,
        description="this is a sample endpoint",
        auth_mode="key"
    )
    ml_client.online_endpoints.begin_create_or_update(endpoint).wait()
    endpoint = ml_client.online_endpoints.get(name=endpoint_name)
    return endpoint


def _deploy_fake_model(ml_client, endpoint_name, deployment_name):
    model = Model(path=os.path.join(Constants.REFERENCES_DIR, "fake_model"))

    deployment = ManagedOnlineDeployment(
        name=deployment_name,
        endpoint_name=endpoint_name,
        model=model,
        environment="azureml://registries/azureml/environments/sklearn-1.0/versions/14",
        code_configuration=CodeConfiguration(
            code=Constants.REFERENCES_DIR, scoring_script="score.py"
        ),
        instance_type="Standard_E2s_v3",
        instance_count=1,
    )
    ml_client.online_deployments.begin_create_or_update(deployment=deployment).wait()
    return deployment


def deploy_fake_test_endpoint_maybe(
        ml_client, endpoint_name="aml-benchmark-test-bzvkqd", deployment_name="test-model",
        use_workspace_name=True,
        connections_name='aml-benchmark-connection'
):
    """Deploy a fake test endpoint."""
    should_deploy = False
    should_wait = True
    endpoint = None
    if use_workspace_name:
        # Endpoint names can be at max 32 characters long.
        endpoint_name = f"{endpoint_name}-{ml_client.workspace_name.split('-')[-1]}"[:32]
    try:
        while should_wait:
            endpoint = ml_client.online_endpoints.get(name=endpoint_name)
            if endpoint.provisioning_state.lower() in ["creating", "updating", 'deleting', 'provisioning']:
                print("Found endpoint in transitional state.")
                time.sleep(30)
                continue
            deployment = ml_client.online_deployments.get(
                endpoint_name=endpoint_name, name=deployment_name)
            if deployment.provisioning_state.lower() == 'failed':
                print("Found endpoint in the failed state, removing.")
                ml_client.online_endpoints.begin_delete(name=endpoint_name).wait()
                should_deploy = True
                break
            elif deployment.provisioning_state.lower() == 'succeeded':
                break
            else:
                time.sleep(30)
                continue
    except Exception as e:
        print(f"Get endpont met {e}")
        should_deploy = True

    if should_deploy:
        try:
            endpoint = _deploy_endpoint(ml_client, endpoint_name)
            deployment = _deploy_fake_model(ml_client, endpoint_name, deployment_name)
        except Exception as e:
            print("Failed deployment due to {}.".format(e))
            print("Trying deploy using a new name now.")
            if "There is already an endpoint with this name" in str(e):
                endpoint_name = f"aml-benchmark-test-{str(uuid.uuid4().hex)}"[:32]
                print("deploying using {}".format(endpoint_name))
                endpoint = _deploy_endpoint(ml_client, endpoint_name)
                deployment = _deploy_fake_model(ml_client, endpoint_name, deployment_name)
            else:
                raise
    if endpoint is None:
        endpoint = ml_client.online_endpoints.get(name=endpoint_name)
    wps_connection = WorkspaceConnection(
        name=connections_name,
        type="azure_sql_db",
        target=endpoint.scoring_uri,
        credentials=AccessKeyConfiguration(
            access_key_id="Authorization",
            secret_access_key=ml_client.online_endpoints.get_keys(endpoint_name).primary_key
        )
    )
    ml_client.connections.create_or_update(workspace_connection=wps_connection)

    return endpoint.scoring_uri, deployment.name, connections_name

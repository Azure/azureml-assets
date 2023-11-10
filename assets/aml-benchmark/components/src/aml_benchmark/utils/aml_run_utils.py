# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Helper functions for AML runs."""
from typing import List, Optional, Tuple, cast, Dict
import os
import tempfile
import re
import json
import requests
from requests.adapters import HTTPAdapter
from urllib3 import Retry

import mlflow
from mlflow.entities import Run as MLFlowRun
from azureml.core import Run
from aml_benchmark.utils.logging import get_logger
from azureml.core import Workspace

from azureml._common._error_definition.azureml_error import AzureMLError
from .exceptions import BenchmarkValidationException
from .error_definitions import BenchmarkValidationError


logger = get_logger(__name__)


def get_experiment_name() -> str:
    """Get the current experiment name."""
    return Run.get_context().experiment.name


def get_parent_run_id() -> str:
    """Get the run id of the parent of the current run."""
    return cast(str, Run.get_context().parent.id)


def get_all_runs_in_current_experiment() -> List[MLFlowRun]:
    """
    Get a list of all of the runs in the current experiment \
    that are a direct child of the root run except the current run.

    :returns: The list of runs in the current experiment.
    """
    experiment_name = get_experiment_name()
    parent_run_id = get_parent_run_id()
    runs = cast(List[MLFlowRun], mlflow.search_runs(
        experiment_names=[experiment_name],
        filter_string=f"tags.mlflow.parentRunId='{parent_run_id}'",
        output_format='list'
    ))
    return [
        run for run in runs
        if run.info.run_id != Run.get_context().id and run.info.run_id != parent_run_id
    ]


def get_compute_information(log_files: Optional[List[str]], run_v1: Run) -> Optional[str]:
    """Get the VMType used for a given run."""
    if log_files is not None:
        file_name = 'execution-wrapper.log'
        complete_file_name = 'system_logs/lifecycler/' + file_name
        if complete_file_name in log_files:
            try:
                with tempfile.TemporaryDirectory() as tmp:
                    run_v1.download_file(complete_file_name, tmp)
                    with open(os.path.join(tmp, file_name), 'r') as file:
                        content = file.read()
                        compute_match = re.search(
                            r'vm_size: Some\("([^\"]*)\"',
                            content,
                        )
                if compute_match:
                    return compute_match.group(1)
            except Exception as ex:
                logger.warn(f"Failed to get system_logs/lifecycler/execution-wrapper.log due to {ex}")
                return None


def get_step_name(run: MLFlowRun) -> str:
    """Get the step name of a given run."""
    stepName = 'stepName'
    if stepName in run.data.tags:
        return run.data.tags[stepName]
    return run.info.run_name


def get_mlflow_model_name_version(model_uri: str) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Get the mlflow model name and version from a URI.

    :param model_uri: the URI from which the model name and version has to be parsed.

    :returns: The model name and version.
    """
    # If the model is from a registry, parse the information from URI
    if model_uri.startswith("azureml://registries/"):
        model_name = model_uri.split('/')[-3]
        model_version = model_uri.split('/')[-1]
        model_registry = model_uri.split('/')[3]
    else:
        model_name = model_uri
        model_version = None
        model_registry = None
    return model_name, model_version, model_registry


def str2bool(v):
    """Convert str to bool."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise BenchmarkValidationException._with_error(
            AzureMLError.create(BenchmarkValidationError, error_details='Boolean value expected.')
        )


def _get_retry_policy(num_retry: int = 3) -> Retry:
    """
    Request retry policy with increasing backoff.

    :return: Returns the msrest or requests REST client retry policy.
    :rtype: urllib3.Retry
    """
    status_forcelist = [413, 429, 500, 502, 503, 504]
    backoff_factor = 0.4
    retry_policy = Retry(
        total=num_retry,
        read=num_retry,
        connect=num_retry,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        # By default this is True. We set it to false to get the full error trace, including url and
        # status code of the last retry. Otherwise, the error message is too many 500 error responses',
        # which is not useful.
        raise_on_status=False
    )
    return retry_policy


def _create_session_with_retry(retry: int = 3) -> requests.Session:
    """
    Create requests.session with retry.

    :type retry: int
    rtype: Response
    """
    retry_policy = _get_retry_policy(num_retry=retry)

    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry_policy))
    session.mount("http://", HTTPAdapter(max_retries=retry_policy))
    return session


def get_authorization_header(connections_name: str) -> Dict[str, str]:
    """Get the api key."""

    def _send_post_request(url: str, headers: dict, payload: dict):
        """Send a POST request."""
        with _create_session_with_retry() as session:
            response = session.post(url, data=json.dumps(payload), headers=headers)
            # Raise an exception if the response contains an HTTP error status code
            response.raise_for_status()

        return response

    try:
        workspace = Run.get_context().experiment.workspace
    except AttributeError:
        # local offline run
        workspace = Workspace.from_config()
    if hasattr(workspace._auth, "get_token"):
        bearer_token = workspace._auth.get_token("https://management.azure.com/.default").token
    else:
        bearer_token = workspace._auth.token

    endpoint = workspace.service_context._get_endpoint("api")
    url_list = [
        endpoint,
        "rp/workspaces/subscriptions",
        workspace.subscription_id,
        "resourcegroups",
        workspace.resource_group,
        "providers",
        "Microsoft.MachineLearningServices",
        "workspaces",
        workspace.name,
        "connections",
        connections_name,
        "listsecrets?api-version=2023-02-01-preview"
    ]
    resp = _send_post_request('/'.join(url_list), {
        "Authorization": f"Bearer {bearer_token}",
        "content-type": "application/json"
    }, {})

    response = resp.json()

    credentials = response['properties'].get('credentials')
    access_key_id = credentials.get('access_key_id')
    if 'secretAccessKey' not in credentials and 'keys' in credentials:
        credentials = credentials['keys']
    if access_key_id == 'api-key':
        token = credentials['secretAccessKey'] 
    else:
        token = 'Bearer ' + credentials['secretAccessKey']
    return {access_key_id if access_key_id else "Authorization": token}

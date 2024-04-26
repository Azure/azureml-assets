# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Helper functions."""

import re
import json
import time
import sys
import subprocess
import requests
from requests.adapters import HTTPAdapter
from urllib3 import Retry
from typing import Callable, Tuple, Optional

from azureml.core import Run, Workspace
from azureml.core.run import _OfflineRun
from azureml._common._error_definition.azureml_error import AzureMLError

from .logging import get_logger
from .constants import Constants
from .exceptions import BenchmarkUserException
from .error_definitions import BenchmarkUserError


logger = get_logger(__name__)


def _get_status_code(e: Exception) -> Optional[int]:
    """
    Get the status code from the exception.

    :param e: Exception.
    :return: Status code.
    """
    status_code = getattr(e, "status_code", None)
    if status_code is None and getattr(e, "response", None) is not None:
        status_code = getattr(e.response, "status_code", None)
    return status_code


def _get_retry_delay(e: Exception) -> Optional[int]:
    """
    Get the retry delay from the exception.

    :param e: Exception.
    :return: Retry delay.
    """
    message = getattr(e, "message", None)
    if message is None:
        # oss does not return this information
        retry_delay = None
    else:
        # oai as well as aoai return this information
        try:
            retry_after = re.search(r"retry after (\d+) second", message)
            retry_delay = int(retry_after.group(1)) + 5
        except Exception:
            retry_delay = None
    return retry_delay


def exponential_backoff(
    max_retries: int = Constants.MAX_RETRIES,
    base_delay: int = Constants.BASE_DELAY,
    max_delay: int = Constants.MAX_DELAY,
    backoff_factor: int = Constants.BACKOFF_FACTOR,
) -> Callable:
    """
    Implement exponential backoff for retrying a function. Use this function as a decorator.

    :prama max_retries: Maximum number of retries.
    :param base_delay: Base delay in seconds before the first retry.
    :param max_delay: Maximum delay in seconds between retries.
    :return: Decorated function.

    Example:
    @exponential_backoff(max_retries=5, base_delay=10, max_delay=60)
    def my_function(...):
        ...
    """

    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            retries = 0
            delay = base_delay
            while retries <= max_retries:
                try:
                    tick = time.time()
                    return func(*args, **kwargs)
                except Exception as e:
                    tock = time.time()
                    status_code = _get_status_code(e)
                    if status_code not in Constants.RETRIABLE_STATUS_CODES:
                        raise
                    retries += 1
                    if retries <= max_retries:
                        backoff_delay = min(delay, max_delay)
                        logger.info(
                            (
                                f"Retrying method `{func.__name__}` after {backoff_delay} sec. "
                                f"Retry attempt: {retries}/{max_retries}. "
                                f"Time spent: {round(tock - tick)} sec. "
                                f"Error details: {e}"
                            )
                        )
                        time.sleep(backoff_delay)
                        delay *= backoff_factor
                    else:
                        raise

        return wrapper

    return decorator


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


def _get_retry_policy(num_retry: int = 3) -> Retry:
    """
    Request retry policy with increasing backoff.

    :return: Returns the msrest or requests REST client retry policy.
    :rtype: urllib3.Retry
    """
    backoff_factor = 0.4
    retry_policy = Retry(
        total=num_retry,
        read=num_retry,
        connect=num_retry,
        backoff_factor=backoff_factor,
        status_forcelist=Constants.RETRIABLE_STATUS_CODES,
        # By default this is True. We set it to false to get the full error trace, including url and
        # status code of the last retry. Otherwise, the error message is too many 500 error responses',
        # which is not useful.
        raise_on_status=False,
    )
    return retry_policy


def _send_post_request(url: str, headers: dict, payload: dict):
    """Send a POST request."""
    try:
        with _create_session_with_retry() as session:
            response = session.post(url, data=json.dumps(payload), headers=headers)
            # Raise an exception if the response contains an HTTP error status code
            response.raise_for_status()
    except requests.exceptions.HTTPError as errh:
        raise BenchmarkUserException._with_error(
            AzureMLError.create(BenchmarkUserError, error_details=f"HTTP Error: {errh}")
        )
    return response


def get_api_key_from_connection(connections_name: str) -> Tuple[str, Optional[str]]:
    """
    Get api_key from connections_name.

    :param connections_name: Name of the connection.
    :return: api_key, api_version.
    """
    run = Run.get_context()
    if isinstance(run, _OfflineRun):
        curr_ws = Workspace.from_config()
    else:
        curr_ws = run.experiment.workspace

    if hasattr(curr_ws._auth, "get_token"):
        bearer_token = curr_ws._auth.get_token(
            "https://management.azure.com/.default"
        ).token
    else:
        bearer_token = curr_ws._auth.token

    endpoint = curr_ws.service_context._get_endpoint("api")
    url_list = [
        endpoint,
        "rp/workspaces/subscriptions",
        curr_ws.subscription_id,
        "resourcegroups",
        curr_ws.resource_group,
        "providers",
        "Microsoft.MachineLearningServices",
        "workspaces",
        curr_ws.name,
        "connections",
        connections_name,
        "listsecrets?api-version=2023-02-01-preview",
    ]

    resp = _send_post_request(
        "/".join(url_list),
        {"Authorization": f"Bearer {bearer_token}", "content-type": "application/json"},
        {},
    )

    credentials = resp.json()["properties"]["credentials"]
    metadata = resp.json()["properties"].get("metadata", {})
    if "key" in credentials:
        return credentials["key"], metadata.get("ApiVersion")
    else:
        if "secretAccessKey" not in credentials and "keys" in credentials:
            credentials = credentials["keys"]
        return credentials["secretAccessKey"], None


def run_command(command: str) -> None:
    """
    Run the command in the shell.

    :param command: The command to run.
    :return: None
    """
    result = subprocess.run(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding=sys.stdout.encoding,
        universal_newlines=True,
        errors="ignore",
    )
    if result.returncode != 0:
        mssg = f"Command: `{command}` failed with error:\n{result.stdout}"
        raise BenchmarkUserException._with_error(
            AzureMLError.create(BenchmarkUserError, error_details=mssg)
        )

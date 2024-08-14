# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Data generator utils."""

import os
import time

from abc import ABC, abstractmethod
from azure.ai.ml import MLClient
from azure.ai.ml.identity import AzureMLOnBehalfOfCredential
from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment, ServerlessEndpoint
from azure.identity import AzureCliCredential, ManagedIdentityCredential
from azureml.acft.common_components import get_logger_app
from azureml.core import Run, Workspace
from azureml.core.run import _OfflineRun
from typing import List, Tuple, Union
from timeit import default_timer as timer

from common.constants import (
    REQUESTS_RETRY_DELAY,
    REGISTRY_MODEL_PATTERN,
    SUPPORTED_STUDENT_MODEL_MAP,
    SUPPORTED_TEACHER_MODEL_MAP,
)


logger = get_logger_app("azureml.acft.contrib.hf.nlp.entry_point.data_import.data_import")
current_run: Run = Run.get_context()


def retry(times: int):
    """Retry utility to wrap.

    Args:
        times (int): No of retries
    """

    def decorator(func):
        def newfn(*args, **kwargs):
            attempt = 1
            while attempt <= times:
                try:
                    return func(*args, **kwargs)
                except Exception:
                    attempt += 1
                    ex_msg = "Exception thrown when attempting to run {}, attempt {} of {}".format(
                        func.__name__, attempt, times
                    )
                    logger.warning(ex_msg)
                    if attempt < times:
                        time.sleep(REQUESTS_RETRY_DELAY)
                    else:
                        logger.warning(
                            "Retried {} times when calling {}, now giving up!".format(times, func.__name__)
                        )
                        raise

        return newfn

    return decorator


def get_credential() -> Union[ManagedIdentityCredential, AzureMLOnBehalfOfCredential]:
    """Create and validate credentials."""
    # try msi, followed by obo, followed by azure cli
    credential = None
    try:
        msi_client_id = os.environ.get("DEFAULT_IDENTITY_CLIENT_ID")
        credential = ManagedIdentityCredential(client_id=msi_client_id)
        credential.get_token("https://management.azure.com/.default")
        logger.info("Using MSI creds")
        return credential
    except Exception:
        logger.error("MSI auth failed")
    try:
        credential = AzureMLOnBehalfOfCredential()
        credential.get_token("https://management.azure.com/.default")
        logger.info("Using OBO creds")
        return credential
    except Exception:
        logger.error("OBO cred failed")
    try:
        credential = AzureCliCredential()
        credential.get_token("https://management.azure.com/.default")
        logger.info("Using OBO creds")
        return credential
    except Exception:
        logger.error("Azure CLI cred failed")

    raise Exception("Error creating credentials.")


def get_workspace() -> Workspace:
    """Return current workspace."""
    if isinstance(current_run, _OfflineRun):
        ws: Workspace = Workspace.from_config("config.json")
    else:
        ws: Workspace = current_run.experiment.workspace
    return ws


def get_workspace_mlclient(workspace: Workspace = None) -> MLClient:
    """Return workspace mlclient."""
    credential = get_credential()
    workspace = get_workspace() if workspace is None else workspace
    if credential and workspace:
        return MLClient(
            credential,
            subscription_id=workspace.subscription_id,
            resource_group_name=workspace.resource_group,
            workspace_name=workspace.name
        )
    raise Exception("Error creating MLClient. No credentials or workspace found")


class EndpointDetails(ABC):
    """Base class for endpoint details."""

    @abstractmethod
    def get_endpoint_key(self) -> str:
        """Get endpoint key."""
        raise NotImplementedError()

    @abstractmethod
    def get_endpoint_url(self) -> str:
        """Get endpoint URL."""
        raise NotImplementedError()

    def get_deployed_model_id(self) -> str:
        """Get deployment model asset id."""
        raise NotImplementedError()


class ServerlessEndpointDetails(EndpointDetails):
    """Serverless endpoint details for data generation."""

    def __init__(self, mlclient_ws: MLClient, endpoint_name: str):
        """Initialize endpoint details for serverless endpoint.

        Args:
            mlclient_ws (MLClient): workspace mlclient
            endpoint_name (str): managed online endpoint name

        Raises:
            Exception: if fetching endpoint details fail
            Exception: if endpoint is not in healthy state
        """
        self._mlclient: MLClient = mlclient_ws
        try:
            self._endpoint: ServerlessEndpoint = self._mlclient.serverless_endpoints.get(endpoint_name)
        except Exception as e:
            raise Exception(f"Serverless endpoint fetch details failed with exception: {e}")

        # ensure endpoint is healthy
        logger.info(f"Endpoint provisioning state: {self._endpoint.provisioning_state}")
        if not (self._endpoint.provisioning_state.lower() == "succeeded"):
            raise Exception(f"Endpoint {self._endpoint.name} is unhealthy.")

    def get_endpoint_key(self) -> str:
        """Get endpoint primary key for serverless deployment.

        Raises:
            Exception: if fetching key fails

        Returns:
            str: endpoint primary key for serverless deployment.
        """
        try:
            return self._mlclient.serverless_endpoints.get_keys(self._endpoint.name).primary_key
        except Exception as e:
            raise Exception(f"Failed to get endpoint keys for endpoint: {self._endpoint.name}. Exception: {e}")

    def get_endpoint_url(self) -> str:
        """Get URL for managed online endpoint."""
        return self._endpoint.scoring_uri

    def get_deployed_model_id(self) -> str:
        """Return deployed model id."""
        return self._endpoint.model_id


class OnlineEndpointDetails(EndpointDetails):
    """Online endpoint details for data generation."""

    def __init__(self, mlclient_ws: MLClient, endpoint_name: str):
        """Initialize endpoint details for managed online endpoint.

        Args:
            mlclient_ws (MLClient): workspace mlclient
            endpoint_name (str): managed online endpoint name

        Raises:
            Exception: if fetching endpoint details fail
            Exception: if traffic is not allocated or allocated to more than one endpoint
            Exception: if endpoint or deployment is not in healthy state
        """
        self._mlclient: MLClient = mlclient_ws
        try:
            self._endpoint: ManagedOnlineEndpoint = self._mlclient.online_endpoints.get(endpoint_name)
        except Exception as e:
            raise Exception(f"Online endpoint fetch details failed with exception: {e}")

        all_deployments = self._get_deployments()
        # fetch deployment with 100% traffic
        deployments = [
            deployment
            for deployment in all_deployments if deployment.name in [
                deployment_name
                for deployment_name, traffic_percent in self._endpoint.traffic.items() if traffic_percent == 100
            ]
        ]

        if len(deployments) != 1:
            raise Exception(
                f"Endpoint {self._endpoint.name} does not meet traffic criteria "
                "of one alone deployment with 100% traffic allocation. "
                f"Currently trafic is allocated to {len(self._endpoint.traffic)} deployments"
            )

        self._deployment = deployments[0]
        # ensure endpoint and deployment is healthy
        logger.info(f"Endpoint provisioning state: {self._endpoint.provisioning_state}")
        logger.info(f"Deployment provisioning state: {self._deployment.provisioning_state}")
        if not (self._endpoint.provisioning_state.lower() == "succeeded"
                and self._deployment.provisioning_state.lower() == "succeeded"):
            raise Exception(f"Endpoint {self._endpoint.name} or deployment {self._deployment.name} is unhealthy.")

    def get_endpoint_key(self):
        """Get endpoint primary key for managed online deployment.

        Raises:
            Exception: if fetching key fails

        Returns:
            str: endpoint primary key for managed online deployment.
        """
        try:
            return self._mlclient.online_endpoints.get_keys(self._endpoint.name).primary_key
        except Exception as e:
            raise Exception(f"Failed to get endpoint keys for endpoint: {self._endpoint.name}. Exception: {e}")

    def get_endpoint_url(self) -> str:
        """Get URL for managed online endpoint."""
        return self._endpoint.scoring_uri

    def get_deployed_model_id(self):
        """Return deployed model id."""
        return self._deployment.model

    def _get_deployments(self) -> List[ManagedOnlineDeployment]:
        """Return list of deployment for current endpoint.

        Returns:
            List[ManagedOnlineDeployment]: List of deployments
        """
        try:
            self._deployments: List[ManagedOnlineDeployment] = self._mlclient.online_deployments.list(
                self._endpoint.name)
            return self._deployments
        except Exception as e:
            logger.error(f"Could not fetch deployments for endpoint: {self._endpoint.name}. Exception => {e}")
            return None


def get_endpoint_details(mlclient_ws: MLClient, endpoint_name: str) -> EndpointDetails:
    """Get endpoint details for endpoint created in workspace or ai project.

    Args:
        mlclient_ws (MLClient): workspace mlclient
        endpoint_name (str): endpoint name of the workspace/aiproject endpoint

    Raises:
        Exception: if endpoint details are not found

    Returns:
        EndpointDetails: endpoint details for endpoint name
    """
    logger.info("Checking if endpoint is a serverless deployment")
    try:
        return ServerlessEndpointDetails(mlclient_ws, endpoint_name)
    except Exception as e:
        logger.warning(f"Fetching serverless endpoint details failed with => {e}")

    try:
        return OnlineEndpointDetails(mlclient_ws, endpoint_name)
    except Exception as e:
        logger.warning(f"Fetching serverless endpoint details failed with => {e}")

    raise Exception(
        f"Could not fetch endpoint {endpoint_name} details for online or serverless deployment."
    )


def _get_model_id_from_run_details():
    # add implementation here to get student model id
    pass


def _get_model_details(model_asset_id, supported_model_map) -> Tuple[str, str, str]:
    # try matching registry model pattern
    if match := REGISTRY_MODEL_PATTERN.match(model_asset_id):
        registry, model_name, model_version = match.group("registry"), match.group("model"), match.group("version")
        # check if model_name exists in supported list
        if model_name not in supported_model_map:
            raise Exception(
                f"Unsupported model: {model_name} for distillation. "
                f"Supported models are: {list(SUPPORTED_TEACHER_MODEL_MAP.keys())}"
            )
        model_details = supported_model_map[model_name]
        supported_registries = model_details["supported_registries"]
        if registry not in supported_registries:
            raise Exception(
                f"Unsupported model registry name: {registry} for model {model_name}. "
                f"Supported registries are: {supported_registries}."
            )
        supported_version_pattern = model_details["supported_version_pattern"]
        if model_version and not supported_version_pattern.match(model_version):
            raise Exception(
                f"Unsupported model version: {model_version} for model {model_name}. "
                f"Model version must match with pattern `{supported_version_pattern}`"
            )
        return registry, model_name, model_version
    raise Exception(
        f"`{model_asset_id}` does not match with model registry pattern. "
        "Please ensure that model in registry is used for teacher and student model combination."
    )


def validate_teacher_model_details(model_asset_id: str) -> Tuple[str, str, str]:
    """Validate and get teacher model details.

    Args:
        model_asset_id (str): registry model asset id

    Returns:
        Tuple[str, str, str]: Tuple containing registry name, model name and model version
    """
    return _get_model_details(model_asset_id, SUPPORTED_TEACHER_MODEL_MAP)


def validate_student_model_details(model_asset_id: str) -> Tuple[str, str, str]:
    """Validate and get student model details.

    Args:
        model_asset_id (str): registry model asset id

    Returns:
        Tuple[str, str, str]: Tuple containing registry name, model name and model version
    """
    return _get_model_details(model_asset_id, SUPPORTED_STUDENT_MODEL_MAP)

def format_time(sec):
    if sec < 1e-6:
        return '%8.2f ns' % (sec * 1e9)
    elif sec < 1e-3:
        return '%8.2f mks' % (sec * 1e6)
    elif sec < 1:
        return '%8.2f ms' % (sec * 1e3)
    else:
        return '%8.2f s' % sec

class LogDuration(object):
    """Times each function call or block execution.
    
    This class provides a mechanism to measure and log the duration of code execution.
    It can be used as a context manager to time a block of code and print the elapsed time
    once the block is exited, or as a decorator to time the entire function call.
    """

    def __init__(self, print_func, label=None, unit='auto', threshold=-1, repr_len=25):
        """Initialize timer parameters.
        
        Args:
            print_func (callable): A function that will be called to print the duration message.
            label (str, optional): An optional label to prepend to the duration message.
            unit (str): The unit of time for displaying the duration. Options are 'ns', 'mks',
                        'ms', 's' and 'auto'.
            threshold (float): Minimum duration in seconds to trigger printing.
            repr_len (int): The maximum length of the string represenation of the duration.
        
        Raises:
            ValueError: If an unknown time unit is provided.
        
        Example:
            >>> def log(msg):
            ...     print(msg)
            >>> with LogDuration(print_func=logger, label="My block") as timer:
            ...     # Your code here
            >>> @LogDuration(print_func=logger, label="My function")
            ...     def foo():
            ...         # Your code here
        """
        self.print_func = print_func
        self.label = label
        if unit not in self.TIME_FORMATS:
            raise ValueError('Unknown time unit: %s. It should be ns, mks, ms, s or auto.' % unit)
        self.format_time = self.TIME_FORMATS[unit]
        self.threshold = threshold

    TIME_FORMATS = {
        'auto': format_time,
        'ns': lambda sec: '%8.2f ns' % (sec * 1e9),
        'mks': lambda sec: '%8.2f mks' % (sec * 1e6),
        'ms': lambda sec: '%8.2f ms' % (sec * 1e3),
        's': lambda sec: '%8.2f s' % sec,
    }

    def __enter__(self):
        """Start the timer when entering the context manager."""
        self.start = timer()
        return self

    def __exit__(self, *exc):
        """Stop the timer and print the duration if it exceeds the threshold."""
        duration = timer() - self.start
        if duration >= self.threshold:
            duration_str = self.format_time(duration)
            self.print_func("%s : Completed in %s" % (self.label, duration_str) if self.label else duration_str)

    def __call__(self, func):
        """Wrap the function with timing functionality."""
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            if duration >= self.threshold:
                duration_str = self.format_time(duration)
                self.print_func("%s : Completed in %s" % (self.label, duration_str) if self.label else duration_str)
            return result
        return wrapper
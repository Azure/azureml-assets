# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains utilities to post a warning event to the current job."""


import logging
import mlflow
from azureml._restclient.models import BaseEvent
from azureml._restclient.run_client import RunClient
from azureml._restclient.service_context import ServiceContext
from azureml.core.authentication import AzureMLTokenAuthentication
from mlflow import MlflowClient

DATE_TIME_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"
MODEL_MONITOR = "Model Monitor"
WARNING_EVENT_NAME = "Microsoft.MachineLearning.Run.Warning"
ROOT_RUN_ID_TAG = "mlflow.rootRunId"
AML_SEND_EMAIL_ADDRESSES_TAG = "azureml.notification.email.address"


def _get_environment_variable_value(key, logger):
    import os

    value = os.environ.get(key)

    if value is None:
        logger.error(f"Failed to get value of environment variable '{key}'.")

    return value


def _get_experiment_name(mlflow_client, experiment_id):
    experiment = mlflow_client.get_experiment(experiment_id)
    return experiment.name


def _get_run_id():
    logger = logging.getLogger("event_utils")
    return _get_environment_variable_value("MLFLOW_RUN_ID", logger)


def _get_service_context(mlflow_client, tracking_token, logger):
    client_service_context = mlflow_client._tracking_client.store.service_context

    return ServiceContext(
        client_service_context._subscription_id,
        client_service_context._resource_group_name,
        client_service_context._workspace_name,
        workspace_id=None,
        workspace_discovery_url=None,
        authentication=AzureMLTokenAuthentication(tracking_token),
    )


def _get_warning_event(run_id, message):
    time = _get_timestamp()
    event_data = {"RunId": run_id, "Source": MODEL_MONITOR, "Message": message}

    return BaseEvent(timestamp=time, name=WARNING_EVENT_NAME, data=event_data)


def _get_timestamp():
    import datetime
    import pytz

    now = datetime.datetime.utcnow().replace(tzinfo=pytz.utc)
    return now.strftime(DATE_TIME_FORMAT)


def add_tags_to_run(run_id: str, tags: dict):
    """Add tags to the run with the specified run id."""
    client = mlflow.tracking.MlflowClient()
    for key, value in tags.items():
        client.set_tag(run_id, key, value)


def add_tags_to_root_run(tags: dict):
    """Add tags to the root run."""
    run = mlflow.get_run(run_id=_get_run_id())
    root_run_id = run.data.tags.get(ROOT_RUN_ID_TAG, None) or run.info.run_id
    add_tags_to_run(root_run_id, tags)


def post_email_event(signal_name: str, emails: str, message: str):
    """Post an email event with message to the current job."""
    run = mlflow.get_run(run_id=_get_run_id())
    root_run_id = run.data.tags.get(ROOT_RUN_ID_TAG, None) or run.info.run_id
    print(f"Posting email event with message '{message}' to run '{root_run_id}'.")
    add_tags_to_run(
        root_run_id,
        {
            AML_SEND_EMAIL_ADDRESSES_TAG: emails,
            f"azureml.modelmonitor.threshold.breached.{signal_name}": message
        })


def post_warning_event(message):
    """Post a warning event with message to the current job.

    Args:
        message: The warning message to be posted.
    """
    logger = logging.getLogger("event_utils")
    logger.warning(f"Warning message: {message}")

    run_id = _get_run_id()
    if run_id is None:
        return

    experiment_id = _get_environment_variable_value("MLFLOW_EXPERIMENT_ID", logger)
    if experiment_id is None:
        return

    mlflow_client = MlflowClient()
    experiment_name = _get_experiment_name(mlflow_client, experiment_id)
    if experiment_name is None:
        logger.error(f"Failed to get name of experiment '{experiment_id}'.")
        return

    tracking_token = _get_environment_variable_value("MLFLOW_TRACKING_TOKEN", logger)
    if tracking_token is None:
        return

    service_context = _get_service_context(mlflow_client, tracking_token, logger)
    if service_context is None:
        logger.error(
            f"Failed to get service context for the resource with tracking uri: '{mlflow.get_tracking_uri()}'."
        )
        return

    event_body = _get_warning_event(run_id, message)

    try:
        client = RunClient(
            service_context, experiment_name, run_id, experiment_id=experiment_id
        )
        client.post_event(event_body)

    except Exception as e:
        logger.error(f"Failed to post a warning event with error: {repr(e)}")
        return

    logger.info("Successfully posted a warning event.")

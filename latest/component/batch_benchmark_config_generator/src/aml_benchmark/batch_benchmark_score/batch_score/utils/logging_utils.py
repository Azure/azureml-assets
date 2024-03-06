# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""The logging utitlities for batch score."""

import argparse
import json
import os
import logging
import time
from .common import constants
from azureml.core import Run
from contextvars import ContextVar
from .events_client import EventsClient, AppInsightsEventsClient
from opencensus.ext.azure.log_exporter import AzureLogHandler


_default_logger_name = "BatchScoreComponent"
_custom_dimensions = {}
_default_logger: logging.LoggerAdapter = None
_events_client: EventsClient = None

_ctx_worker_id = ContextVar("Async worker ID", default=None)
_ctx_mini_batch_id = ContextVar("Async mini-batch ID", default=None)
_ctx_quota_audience = ContextVar("Async quota audience", default=None)
_ctx_batch_pool = ContextVar("Async batch pool", default=None)


class UTCFormatter(logging.Formatter):
    """Class for UTC formatter."""

    converter = time.gmtime


def setup_logger(log_level: str, app_insights_connection_string: str = None):
    """Init logger."""
    global _custom_dimensions
    global _default_logger
    global _events_client

    _custom_dimensions = __calculate_custom_dimensions()

    stream_formatter = UTCFormatter('%(asctime)s - %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(stream_formatter)

    logger = logging.getLogger(_default_logger_name)

    logger.addHandler(stream_handler)

    if app_insights_connection_string is not None:
        print("Enabling application insights logs")
        azure_formatter = logging.Formatter('%(message)s')
        azure_handler = AzureLogHandler(connection_string=app_insights_connection_string)
        azure_handler.setFormatter(azure_formatter)
        logger.addHandler(azure_handler)

    logger.setLevel(log_level)

    _default_logger = logger

    if app_insights_connection_string is not None:
        _events_client = AppInsightsEventsClient(
            _custom_dimensions, app_insights_connection_string,
            _ctx_worker_id, _ctx_mini_batch_id, _ctx_quota_audience, _ctx_batch_pool)
    else:
        _events_client = EventsClient()


def get_logger():
    """Get logger."""
    custom_dimensions = _custom_dimensions.copy()
    custom_dimensions["WorkerId"] = _ctx_worker_id.get()
    custom_dimensions["MiniBatchId"] = _ctx_mini_batch_id.get()
    custom_dimensions["QuotaAudience"] = _ctx_quota_audience.get()
    custom_dimensions["BatchPool"] = _ctx_batch_pool.get()

    extra = {
        "custom_dimensions": custom_dimensions,
    }

    return logging.LoggerAdapter(_default_logger, extra)


def get_events_client():
    """Get events client."""
    return _events_client


def set_worker_id(worker_id: int):
    """Set worker id."""
    _ctx_worker_id.set(worker_id)


def set_mini_batch_id(mini_batch_id: int):
    """Set mini batch id."""
    _ctx_mini_batch_id.set(mini_batch_id)


def set_quota_audience(quota_audience: str):
    """Set quota audience."""
    _ctx_quota_audience.set(quota_audience)


def set_batch_pool(batch_pool: str):
    """Set batch pool."""
    _ctx_batch_pool.set(batch_pool)


def __calculate_custom_dimensions():
    custom_dimensions = {}
    custom_dimensions["HostName"] = os.environ.get(constants.OS_ENVIRON_HOST_NAME, "Unknown")
    custom_dimensions["Workspace"] = os.environ.get(constants.OS_ENVIRON_WORKSPACE, "Unknown")
    custom_dimensions["RunId"] = os.environ.get(constants.OS_ENVIRON_RUN_ID, "Unknown")
    custom_dimensions["NodeRank"] = os.environ.get(constants.OS_ENVIRON_NODE_RANK, "Unknown")
    custom_dimensions["ComponentVersion"] = constants.BATCH_SCORE_COMPONENT_VERSION

    try:
        run = Run.get_context()
        custom_dimensions["ParentRunId"] = run.parent.id
        custom_dimensions["StepName"] = run.name
        custom_dimensions["ExperimentName"] = run.experiment.name
    except Exception as e:
        print(e)
        print("Failed to get run context")

    try:
        prs_internal_arg_name = "bf89b5b0_523f_4782_aedc_61bd625ee81a"
        parser = argparse.ArgumentParser()
        parser.add_argument(f"--{prs_internal_arg_name}", nargs='?', const=None, type=str)
        args, unknown_args = parser.parse_known_args()

        arg_value = getattr(args, prs_internal_arg_name)
        if arg_value is not None:
            agent_args = json.loads(arg_value)
            custom_dimensions["PRSAgentName"] = agent_args["agent_name"]
    except Exception as e:
        print(e)
        print("Failed to get PRS agent name")

    try:
        compute_context_value = os.environ.get(constants.OS_ENVIRON_COMPUTE_CONTEXT, None)
        if compute_context_value is not None:
            compute_context = json.loads(compute_context_value)
            custom_dimensions["ClusterName"] = compute_context["cluster_name"]
            if "node_id" in compute_context:
                custom_dimensions["NodeId"] = compute_context["node_id"]["Literal"]
    except Exception as e:
        print(e)
        print("Failed to parse compute context")

    return custom_dimensions

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Logging utilities."""

import argparse
import json
import logging
import os
import time
from contextvars import ContextVar

from azureml.core import Run
from opencensus.ext.azure.log_exporter import AzureLogHandler

from .. import constants
from .events_client import AppInsightsEventsClient, EventsClient

_default_logger_name = "BatchScoreComponent"
_custom_dimensions = {}
_default_logger: logging.LoggerAdapter = None
_events_client: EventsClient = None

_ctx_worker_id = ContextVar("Async worker ID", default=None)
_ctx_mini_batch_id = ContextVar("Async mini-batch ID", default=None)
_ctx_quota_audience = ContextVar("Async quota audience", default=None)
_ctx_batch_pool = ContextVar("Async batch pool", default=None)


class UTCFormatter(logging.Formatter):
    """UTC formatter."""

    converter = time.gmtime


class CustomerLogsFilter(logging.Filter):
    """Customer log filter."""

    def __init__(self, log_level: str):
        """Initialize CustomerLogsFilter."""
        self.log_level = logging.getLevelName(log_level)

    def filter(self, record):
        """Filter the record based on the given rule."""
        """
        Customers can see:
         - All logs higher than INFO level, even for internal classes,
         - For INFO and DEBUG, customer sees no logs from internal classes.
         - For all other logs, customer sees their requested log level and higher.
        """
        if record.levelno > logging.INFO:
            return True
        message = record.getMessage()
        for internal_class in [
                'AIMD',
                'Conductor',
                'Gatherer',
                'ParallelDriver',
                'QuotaClient',
                'RoutingClient',
                'WaitTimeCongestionDetector']:
            if message.startswith(internal_class):
                return False
        return record.levelno >= self.log_level


class AppInsightsLogsFilter(logging.Filter):
    """Application insight log filter."""

    def filter(self, record):
        """Prevent logs from going to App Insights by prefacing them with 'AppInsRedact'."""
        message = record.getMessage()
        return not message.startswith('AppInsRedact')


def setup_logger(
        stdout_log_level: str,
        app_insights_log_level: str = None,
        app_insights_connection_string: str = None):
    """Set up logger."""
    global _custom_dimensions
    global _default_logger
    global _events_client

    default_format = '%(asctime)-15s :%(name)-5s:%(levelname)-8s- %(message)s'

    set_default_logger_format(default_format)

    stdout_log_level_upper = stdout_log_level.upper()
    # The root logger has a pre-configured stream handler to log to stdout.
    # A filter is added to clean up the logs sent to customer.
    stream_handler = logging.root.handlers[0]
    stream_handler.addFilter(CustomerLogsFilter(stdout_log_level_upper))

    _custom_dimensions = __calculate_custom_dimensions()

    logger = logging.getLogger(_default_logger_name)

    if app_insights_connection_string is not None:
        app_insights_log_level_upper = app_insights_log_level.upper()
        print(f"Enabling application insights logs, level: {app_insights_log_level_upper}")
        azure_formatter = logging.Formatter('%(message)s')
        azure_handler = AzureLogHandler(connection_string=app_insights_connection_string)
        azure_handler.setFormatter(azure_formatter)
        azure_handler.setLevel(app_insights_log_level_upper)
        azure_handler.addFilter(AppInsightsLogsFilter())
        logger.addHandler(azure_handler)

    logger.setLevel(stdout_log_level_upper)

    _default_logger = logger

    if app_insights_connection_string is not None:
        _events_client = AppInsightsEventsClient(
            _custom_dimensions,
            app_insights_connection_string,
            _ctx_worker_id,
            _ctx_mini_batch_id,
            _ctx_quota_audience,
            _ctx_batch_pool)
    else:
        _events_client = EventsClient()


def set_default_logger_format(default_format, root_handlers=None):
    """Set default logger format."""
    # This overrides the basicConfig on the root logging handler that is writing logs to stdout
    # Root will always be in debug
    log_level = 'DEBUG'

    if root_handlers is None:
        root_handlers = []

    try:
        logging.root.handlers[0].setFormatter(logging.Formatter(default_format))
        logging.root.handlers[0].setLevel(log_level)
    except IndexError:
        # If basic config has not been configured
        logging.basicConfig(format=default_format, level=log_level)
    logging.root.handlers.extend(root_handlers)


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


def get_worker_id():
    """Get worker id."""
    return _ctx_worker_id.get()


def set_worker_id(worker_id: int):
    """Set worker id."""
    _ctx_worker_id.set(worker_id)


def get_mini_batch_id():
    """Get mini batch id."""
    return _ctx_mini_batch_id.get()


def set_mini_batch_id(mini_batch_id: int):
    """Set mini batch id."""
    _ctx_mini_batch_id.set(mini_batch_id)


def get_quota_audience():
    """Get quota audience."""
    return _ctx_quota_audience.get()


def set_quota_audience(quota_audience: str):
    """Set quota audience."""
    _ctx_quota_audience.set(quota_audience)


def get_batch_pool():
    """Get batch pool."""
    return _ctx_batch_pool.get()


def set_batch_pool(batch_pool: str):
    """Set batch pool."""
    _ctx_batch_pool.set(batch_pool)


def __calculate_custom_dimensions():
    custom_dimensions = {}
    custom_dimensions["HostName"] = os.environ.get(constants.OS_ENVIRON_HOST_NAME, "Unknown")
    custom_dimensions["Workspace"] = os.environ.get(constants.OS_ENVIRON_WORKSPACE, "Unknown")
    custom_dimensions["RunId"] = os.environ.get(constants.OS_ENVIRON_RUN_ID, "Unknown")
    custom_dimensions["NodeRank"] = os.environ.get(constants.OS_ENVIRON_NODE_RANK, "Unknown")

    try:
        run = Run.get_context()
        custom_dimensions["ParentRunId"] = run.parent.id
        custom_dimensions["StepName"] = run.name
        custom_dimensions["ExperimentName"] = run.experiment.name
    except Exception:
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
    except Exception:
        print("Failed to get PRS agent name")

    try:
        compute_context_value = os.environ.get(constants.OS_ENVIRON_COMPUTE_CONTEXT, None)
        if compute_context_value is not None:
            compute_context = json.loads(compute_context_value)
            custom_dimensions["ClusterName"] = compute_context["cluster_name"]
            if "node_id" in compute_context:
                custom_dimensions["NodeId"] = compute_context["node_id"]["Literal"]
    except Exception:
        print("Failed to parse compute context")

    return custom_dimensions

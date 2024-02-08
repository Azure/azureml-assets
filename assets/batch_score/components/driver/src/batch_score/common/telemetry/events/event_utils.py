# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains the definition for event utils."""

import os
import functools
import traceback

from contextvars import ContextVar
from datetime import datetime
from pydispatch import dispatcher
from strenum import StrEnum

from ..logging_utils import get_logger
from ...configuration.configuration import Configuration
from ...configuration.metadata import Metadata
from ...constants import BATCH_SCORE_SURFACE_TELEMETRY_EXCEPTIONS_ENV_VAR

# TODO: Investigate if this can be common_context_vars that is not restricted for use in events.
_ctx_api_type = ContextVar("Api type", default=None)
_ctx_async_mode = ContextVar("Async mode", default=None)
_ctx_authentication_type = ContextVar("Authentication type", default=None)
_ctx_component_name = ContextVar("Component name", default=None)
_ctx_component_version = ContextVar("Component version", default=None)
_ctx_endpoint_type = ContextVar("Endpoint type", default=None)


def get_api_type():
    """Get the API type."""
    return _ctx_api_type.get()


def get_async_mode():
    """Get if async mode is enabled or not."""
    return _ctx_async_mode.get()


def get_authentication_type():
    """Get the authentication type."""
    return _ctx_authentication_type.get()


def get_component_name():
    """Get the component name."""
    return _ctx_component_name.get()


def get_component_version():
    """Get the component version."""
    return _ctx_component_version.get()


def get_endpoint_type():
    """Get the endpoint type."""
    return _ctx_endpoint_type.get()


def setup_context_vars(configuration: Configuration, metadata: Metadata):
    """Set up the context variables."""
    _ctx_api_type.set(configuration.get_api_type())
    _ctx_async_mode.set(configuration.async_mode)
    _ctx_authentication_type.set(configuration.get_authentication_type())
    _ctx_component_name.set(metadata.component_name)
    _ctx_component_version.set(metadata.component_version)
    _ctx_endpoint_type.set(configuration.get_endpoint_type())


def emit_event(batch_score_event):
    """Emit the event using the dispatcher."""
    try:
        dispatcher.send(batch_score_event=batch_score_event, signal=Signal.EmitTelemetryEvent)
    except Exception:
        # TO DO: Replace get_logger().info with an event to Geneva for debugging
        get_logger().info(f"An exception occurred while emitting event {batch_score_event.name}: "
                          f"{traceback.format_exc()}")
        if os.getenv(BATCH_SCORE_SURFACE_TELEMETRY_EXCEPTIONS_ENV_VAR) == 'True':
            raise


def add_handler(handler, sender=dispatcher.Any, signal=dispatcher.Any):
    """Add the handler to the dispatcher."""
    dispatcher.connect(handler, sender=sender, signal=signal)


def remove_handler(handler, sender=dispatcher.Any, signal=dispatcher.Any):
    """Remove the handler from the dispatcher."""
    dispatcher.disconnect(handler, sender=sender, signal=signal)


def catch_and_log_all_exceptions(f):
    """Catch and log exceptions."""
    @functools.wraps(f)
    def inner(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception:
            # TO DO: Replace get_logger().info with an event to Geneva for debugging
            get_logger().info(f"An exception occurred in {f.__name__}: {traceback.format_exc()}")
            if os.getenv(BATCH_SCORE_SURFACE_TELEMETRY_EXCEPTIONS_ENV_VAR) == 'True':
                raise
            return None
    return inner


class Signal(StrEnum):
    """Defines the signal."""

    EmitTelemetryEvent = 'EmitTelemetryEvent'
    GenerateMinibatchSummary = 'GenerateMinibatchSummary'


@catch_and_log_all_exceptions
def generate_minibatch_summary(
        minibatch_id: str,
        timestamp: datetime = None,
        output_row_count: int = None):
    """Generate the minibatch summary."""
    dispatcher.send(
        signal=Signal.GenerateMinibatchSummary,
        minibatch_id=minibatch_id,
        timestamp=timestamp or datetime.now(),
        output_row_count=output_row_count or 0,
    )

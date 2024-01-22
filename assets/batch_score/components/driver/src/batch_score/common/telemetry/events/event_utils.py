# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from contextvars import ContextVar
from datetime import datetime
from pydispatch import dispatcher
from strenum import StrEnum

from ...configuration.configuration import Configuration
from ...configuration.metadata import Metadata

# TODO: Investigate if this can be common_context_vars that is not restricted for use in events.
_ctx_api_type = ContextVar("Api type", default=None)
_ctx_async_mode = ContextVar("Async mode", default=None)
_ctx_authentication_type = ContextVar("Authentication type", default=None)
_ctx_component_name = ContextVar("Component name", default=None)
_ctx_component_version = ContextVar("Component version", default=None)
_ctx_endpoint_type = ContextVar("Endpoint type", default=None)

def get_api_type():
    return _ctx_api_type.get()

def get_async_mode():
    return _ctx_async_mode.get()

def get_authentication_type():
    return _ctx_authentication_type.get()

def get_component_name():
    return _ctx_component_name.get()

def get_component_version():
    return _ctx_component_version.get()

def get_endpoint_type():
    return _ctx_endpoint_type.get()

def setup_context_vars(configuration: Configuration, metadata: Metadata):
    _ctx_api_type.set(configuration.get_api_type())
    _ctx_async_mode.set(configuration.async_mode)
    _ctx_authentication_type.set(configuration.get_authentication_type())
    _ctx_component_name.set(metadata.component_name)
    _ctx_component_version.set(metadata.component_version)
    _ctx_endpoint_type.set(configuration.get_endpoint_type())

def emit_event(batch_score_event):
    try:
        dispatcher.send(batch_score_event=batch_score_event)
    except:
        # TBD: Handle exceptions without exposing to the user (write to Geneva for debugging?)
        pass

def add_handler(handler, sender=dispatcher.Any, signal=dispatcher.Any):
    dispatcher.connect(handler, sender=sender, signal=signal)

def remove_handler(handler, sender=dispatcher.Any, signal=dispatcher.Any):
    dispatcher.disconnect(handler, sender=sender, signal=signal)

class Signal(StrEnum):
    GenerateMinibatchSummary = 'GenerateMinibatchSummary'

def generate_minibatch_summary(
    minibatch_id: str,
    timestamp: datetime = None,
    output_row_count: int = None,
):
    try:
        dispatcher.send(
            signal=Signal.GenerateMinibatchSummary,
            minibatch_id=minibatch_id,
            timestamp=timestamp or datetime.now(),
            output_row_count=output_row_count or 0,
        )
    except:
        # TBD: Handle exceptions without exposing to the user (write to Geneva for debugging?)
        pass
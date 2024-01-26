# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Geneva event listener."""

from ..events.batch_score_event import BatchScoreEvent
from ..events.batch_score_init_completed_event import BatchScoreInitCompletedEvent
from ..events.batch_score_init_started_event import BatchScoreInitStartedEvent
from ..events.batch_score_minibatch_completed_event import BatchScoreMinibatchCompletedEvent
from ..events.batch_score_minibatch_started_event import BatchScoreMinibatchStartedEvent
from ..events.event_utils import add_handler, catch_and_log_all_exceptions, Signal
from ..geneva_event_client import GenevaEventClient


def setup_geneva_event_handlers():
    """Set up Geneva event handlers."""
    add_handler(_handle_batch_score_event, signal=Signal.EmitTelemetryEvent)


_geneva_event_client = GenevaEventClient()
_geneva_event_types = (
    BatchScoreInitCompletedEvent,
    BatchScoreInitStartedEvent,
    BatchScoreMinibatchCompletedEvent,
    BatchScoreMinibatchStartedEvent,
)


@catch_and_log_all_exceptions
def _handle_batch_score_event(batch_score_event: BatchScoreEvent = None, sender=None, signal=None):
    if batch_score_event is None:
        return

    if isinstance(batch_score_event, _geneva_event_types):
        _geneva_event_client.emit_event(batch_score_event)

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Job log event listener."""

from .. import logging_utils as lu
from ..events.batch_score_event import BatchScoreEvent
from ..events.event_utils import add_handler, catch_and_log_all_exceptions, Signal


@catch_and_log_all_exceptions
def handle_batch_score_event(batch_score_event: BatchScoreEvent = None, sender=None, signal=None):
    """Handle batch score event."""
    # Depending on the event type/other logic, choose info/debug/error/ignore as needed
    if batch_score_event is None:
        return

    lu.get_logger().info(
        f"{str(batch_score_event.event_time)}: {type(batch_score_event).__name__}: {batch_score_event.__str__()}")


def setup_job_log_event_handlers():
    """Set up job log event handlers."""
    add_handler(handle_batch_score_event, signal=Signal.EmitTelemetryEvent)

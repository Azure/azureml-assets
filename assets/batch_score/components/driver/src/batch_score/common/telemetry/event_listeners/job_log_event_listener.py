# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from datetime import datetime, timezone

from .. import logging_utils as lu
from ..events.batch_score_event import BatchScoreEvent
from ..events.event_utils import add_handler

def handle_batch_score_event(batch_score_event: BatchScoreEvent = None):
    # Depending on the event type/other logic, choose info/debug/error/ignore as needed
    if batch_score_event is None:
        return

    event_time = str(datetime.now(timezone.utc))
    lu.get_logger().info(f"{str(batch_score_event.event_time)}: {type(batch_score_event).__name__}: {batch_score_event.__str__()}")

def setup_job_log_event_handlers():
    add_handler(handle_batch_score_event)

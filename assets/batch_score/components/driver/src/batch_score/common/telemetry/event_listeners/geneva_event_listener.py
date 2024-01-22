# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from ..events.batch_score_event import BatchScoreEvent
from ..events.batch_score_init_completed_event import BatchScoreInitCompletedEvent
from ..events.batch_score_init_started_event import BatchScoreInitStartedEvent
from ..events.batch_score_minibatch_completed_event import BatchScoreMinibatchCompletedEvent
from ..events.batch_score_minibatch_started_event import BatchScoreMinibatchStartedEvent
from ..events.event_utils import add_handler
from ..geneva_event_client import GenevaEventClient

def setup_geneva_event_handlers():
    add_handler(_handle_batch_score_event)

_geneva_event_client = GenevaEventClient()
_geneva_event_types = (
    BatchScoreInitCompletedEvent,
    BatchScoreInitStartedEvent,
    BatchScoreMinibatchCompletedEvent,
    BatchScoreMinibatchStartedEvent,
)

def _handle_batch_score_event(batch_score_event: BatchScoreEvent = None):
    if batch_score_event is None:
        return

    if isinstance(batch_score_event, _geneva_event_types):
        _geneva_event_client.emit_event(batch_score_event)
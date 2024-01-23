from datetime import datetime

from ..events.event_utils import (
    add_handler,
    emit_event,
    remove_handler,
    Signal,
)
from ..events.batch_score_event import BatchScoreEvent
from ..events import event_utils
from ..minibatch_aggregator import MinibatchAggregator


def setup_minibatch_aggregator_event_handlers():
    add_handler(_handle_batch_score_event)
    add_handler(_handle_generate_minibatch_summary, signal=Signal.GenerateMinibatchSummary)


def teardown_minibatch_aggregator_event_handlers():
    remove_handler(_handle_batch_score_event)
    remove_handler(_handle_generate_minibatch_summary, signal=Signal.GenerateMinibatchSummary)


_minibatch_aggregator = MinibatchAggregator()


@event_utils.catch_and_log_all_exceptions
def _handle_batch_score_event(batch_score_event: BatchScoreEvent = None, sender=None, signal=None):
    _minibatch_aggregator.add(batch_score_event)


@event_utils.catch_and_log_all_exceptions
def _handle_generate_minibatch_summary(
    signal: Signal = None,
    minibatch_id: str = None,
    timestamp: datetime = None,
    output_row_count: int = None,
    sender=None
):
    summary_event = _minibatch_aggregator.summarize(
        minibatch_id=minibatch_id,
        timestamp=timestamp,
        output_row_count=output_row_count,
    )
    _minibatch_aggregator.clear(minibatch_id=minibatch_id)
    emit_event(batch_score_event=summary_event)

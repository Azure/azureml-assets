# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Minibatch aggregator."""

import numpy as np

from collections import defaultdict
from datetime import datetime

from .events.batch_score_event import BatchScoreEvent
from .events.batch_score_minibatch_completed_event import BatchScoreMinibatchCompletedEvent
from .events.batch_score_minibatch_started_event import BatchScoreMinibatchStartedEvent
from .events.batch_score_request_completed_event import BatchScoreRequestCompletedEvent
from .events.batch_score_input_row_completed_event import BatchScoreInputRowCompletedEvent


class MinibatchAggregator:
    """Minibatch aggregator."""

    def __init__(self):
        """Initialize MinibatchAggregator."""
        self._http_request_completed_events_per_minibatch = defaultdict(list)
        self._rows_completed_events_per_minibatch = defaultdict(list)
        self._start_event_per_minibatch = defaultdict(BatchScoreMinibatchStartedEvent)

    def add(self, event: BatchScoreEvent = None, **kwargs) -> None:
        """Add a batch score event to its minibatch."""
        if isinstance(event, BatchScoreRequestCompletedEvent):
            self._http_request_completed_events_per_minibatch[event.minibatch_id].append(event)

        if isinstance(event, BatchScoreInputRowCompletedEvent):
            self._rows_completed_events_per_minibatch[event.minibatch_id].append(event)

        if isinstance(event, BatchScoreMinibatchStartedEvent):
            self._start_event_per_minibatch[event.minibatch_id] = event

    def clear(self, minibatch_id: str) -> None:
        """Clear the minibatch maps after the aggregation is done."""
        del self._http_request_completed_events_per_minibatch[minibatch_id]
        del self._rows_completed_events_per_minibatch[minibatch_id]
        del self._start_event_per_minibatch[minibatch_id]

    def summarize(
            self,
            minibatch_id: str,
            end_time: datetime,
            output_row_count: int) -> BatchScoreMinibatchCompletedEvent:
        """Summarize the minibatch events into a BatchScoreMinibatchCompletedEvent."""
        http_request_completed_events = self._http_request_completed_events_per_minibatch[minibatch_id]
        http_request_durations_ms = [e.duration_ms for e in http_request_completed_events] or [0]

        minibatch_end_time = end_time.timestamp()
        if minibatch_id in self._start_event_per_minibatch:
            minibatch_start_time = self._start_event_per_minibatch[minibatch_id].event_time.timestamp()
        else:
            minibatch_start_time = minibatch_end_time
        rows_completed_events = self._rows_completed_events_per_minibatch[minibatch_id]
        row_completed_timestamps = sorted(
            e.event_time.timestamp() for e in rows_completed_events) or [minibatch_start_time]

        return BatchScoreMinibatchCompletedEvent(
            minibatch_id=minibatch_id,
            scoring_url=self._start_event_per_minibatch[minibatch_id].scoring_url,
            batch_pool=self._start_event_per_minibatch[minibatch_id].batch_pool,
            quota_audience=self._start_event_per_minibatch[minibatch_id].quota_audience,

            total_prompt_tokens=sum(e.prompt_tokens or 0 for e in rows_completed_events),
            total_completion_tokens=sum(e.completion_tokens or 0 for e in rows_completed_events),

            input_row_count=self._start_event_per_minibatch[minibatch_id].input_row_count or 0,
            succeeded_row_count=len([e for e in rows_completed_events if e.is_successful]),
            failed_row_count=len([e for e in rows_completed_events if not e.is_successful]),
            output_row_count=output_row_count,

            http_request_count=len(http_request_completed_events),
            http_request_succeeded_count=len(
                [e for e in http_request_completed_events if 200 <= e.response_code < 300]),
            http_request_user_error_count=len(
                [e for e in http_request_completed_events if 400 <= e.response_code < 500]),
            http_request_system_error_count=len(
                [e for e in http_request_completed_events if 500 <= e.response_code < 600]),
            http_request_retry_count=sum(e.retry_count for e in rows_completed_events),

            http_request_duration_p0_ms=np.min(http_request_durations_ms),
            http_request_duration_p50_ms=np.percentile(http_request_durations_ms, 50),
            http_request_duration_p90_ms=np.percentile(http_request_durations_ms, 90),
            http_request_duration_p95_ms=np.percentile(http_request_durations_ms, 95),
            http_request_duration_p99_ms=np.percentile(http_request_durations_ms, 99),
            http_request_duration_p100_ms=np.max(http_request_durations_ms),

            progress_duration_p0_ms=1000 * (np.percentile(row_completed_timestamps, 0) - minibatch_start_time),
            progress_duration_p50_ms=1000 * (np.percentile(row_completed_timestamps, 50) - minibatch_start_time),
            progress_duration_p90_ms=1000 * (np.percentile(row_completed_timestamps, 90) - minibatch_start_time),
            progress_duration_p95_ms=1000 * (np.percentile(row_completed_timestamps, 95) - minibatch_start_time),
            progress_duration_p99_ms=1000 * (np.percentile(row_completed_timestamps, 99) - minibatch_start_time),
            progress_duration_p100_ms=1000 * (np.percentile(row_completed_timestamps, 100) - minibatch_start_time),
            total_duration_ms=1000 * (minibatch_end_time - minibatch_start_time),
        )

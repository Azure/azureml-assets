# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Minibatch aggregator."""

import numpy as np

from collections import defaultdict
from datetime import datetime

from .events.batch_score_event import BatchScoreEvent
from .events.batch_score_minibatch_completed_event import BatchScoreMinibatchCompletedEvent
from .events.batch_score_minibatch_endpoint_health_event import BatchScoreMinibatchEndpointHealthEvent
from .events.batch_score_minibatch_started_event import BatchScoreMinibatchStartedEvent
from .events.batch_score_request_completed_event import BatchScoreRequestCompletedEvent
from .events.batch_score_input_row_completed_event import BatchScoreInputRowCompletedEvent

from . import logging_utils as lu


class MinibatchAggregator:
    """Minibatch aggregator."""

    def __init__(self):
        """Initialize MinibatchAggregator."""
        self._http_request_completed_events_per_minibatch = defaultdict(list)
        self._http_request_completed_events_per_minibatch_per_endpoint = defaultdict(lambda: defaultdict(list))
        self._rows_completed_events_per_minibatch = defaultdict(list)
        self._start_event_per_minibatch = defaultdict(BatchScoreMinibatchStartedEvent)
        self._model_name = None
        self._emit_endpoint_health_events = False

    def add(self, event: BatchScoreEvent = None, **kwargs) -> None:
        """Add a batch score event to its minibatch."""
        if isinstance(event, BatchScoreRequestCompletedEvent):
            self._http_request_completed_events_per_minibatch[event.minibatch_id].append(event)
            request_endpoint_map = self._http_request_completed_events_per_minibatch_per_endpoint
            request_endpoint_map[event.minibatch_id][event.scoring_url].append(event)

            if not self._model_name:
                self._model_name = event.model_name

        if isinstance(event, BatchScoreInputRowCompletedEvent):
            self._rows_completed_events_per_minibatch[event.minibatch_id].append(event)

        if isinstance(event, BatchScoreMinibatchStartedEvent):
            self._start_event_per_minibatch[event.minibatch_id] = event
            self._emit_endpoint_health_events = True

    def clear(self, minibatch_id: str) -> None:
        """Clear the minibatch maps after the aggregation is done."""
        if minibatch_id in self._http_request_completed_events_per_minibatch:
            del self._http_request_completed_events_per_minibatch[minibatch_id]

        if minibatch_id in self._http_request_completed_events_per_minibatch_per_endpoint:
            del self._http_request_completed_events_per_minibatch_per_endpoint[minibatch_id]

        if minibatch_id in self._rows_completed_events_per_minibatch:
            del self._rows_completed_events_per_minibatch[minibatch_id]

        if minibatch_id in self._start_event_per_minibatch:
            del self._start_event_per_minibatch[minibatch_id]

        self._model_name = None
        self._emit_endpoint_health_events = False

    def summarize(
            self,
            minibatch_id: str,
            end_time: datetime,
            output_row_count: int,
            logging_metadata: dict = None) -> BatchScoreMinibatchCompletedEvent:
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

        lu.get_logger().info(f"Minibatch {minibatch_id}: Successfully summarized "
                             f"http_request_completed_events: {len(http_request_completed_events)}, "
                             f"rows_completed_events: {len(rows_completed_events)}.")

        return BatchScoreMinibatchCompletedEvent(
            minibatch_id=minibatch_id,
            scoring_url=self._start_event_per_minibatch[minibatch_id].scoring_url,
            quota_audience=self._start_event_per_minibatch[minibatch_id].quota_audience,
            model_name=self._model_name,
            retry_count=self._start_event_per_minibatch[minibatch_id].retry_count,

            total_prompt_tokens=sum(e.prompt_tokens or 0 for e in rows_completed_events),
            total_completion_tokens=sum(e.completion_tokens or 0 for e in rows_completed_events),

            input_row_count=self._start_event_per_minibatch[minibatch_id].input_row_count or 0,
            succeeded_row_count=len([e for e in rows_completed_events if e.is_successful]),
            failed_row_count=len([e for e in rows_completed_events if not e.is_successful]),
            output_row_count=output_row_count,
            logging_metadata=logging_metadata,

            http_request_count=len(http_request_completed_events),
            http_request_succeeded_count=len(
                [e for e in http_request_completed_events if self._is_request_succeeded(e)]),
            http_request_user_error_count=len(
                [e for e in http_request_completed_events if self._is_request_user_error(e)]),
            http_request_system_error_count=len(
                [e for e in http_request_completed_events if self._is_request_system_error(e)]),
            http_request_retry_count=sum(e.retry_count for e in rows_completed_events),

            http_request_duration_p0_ms=float(np.min(http_request_durations_ms)),
            http_request_duration_p50_ms=float(np.percentile(http_request_durations_ms, 50)),
            http_request_duration_p90_ms=float(np.percentile(http_request_durations_ms, 90)),
            http_request_duration_p95_ms=float(np.percentile(http_request_durations_ms, 95)),
            http_request_duration_p99_ms=float(np.percentile(http_request_durations_ms, 99)),
            http_request_duration_p100_ms=float(np.max(http_request_durations_ms)),

            progress_duration_p0_ms=1000 * float(np.percentile(row_completed_timestamps, 0) - minibatch_start_time),
            progress_duration_p50_ms=1000 * float(np.percentile(row_completed_timestamps, 50) - minibatch_start_time),
            progress_duration_p90_ms=1000 * float(np.percentile(row_completed_timestamps, 90) - minibatch_start_time),
            progress_duration_p95_ms=1000 * float(np.percentile(row_completed_timestamps, 95) - minibatch_start_time),
            progress_duration_p99_ms=1000 * float(np.percentile(row_completed_timestamps, 99) - minibatch_start_time),
            progress_duration_p100_ms=1000 * float(np.percentile(row_completed_timestamps, 100)
                                                   - minibatch_start_time),
            total_duration_ms=1000 * (minibatch_end_time - minibatch_start_time),
        )

    def summarize_endpoints(self, minibatch_id: str, logging_metadata: dict = None) -> list:
        """Summarize the minibatch events into a list of BatchScoreMinibatchEndpointHealthEvents."""
        if not self._emit_endpoint_health_events:
            return []

        processed_event_count = 0
        request_endpoint_map = self._http_request_completed_events_per_minibatch_per_endpoint
        http_request_completed_events_per_endpoint = request_endpoint_map[minibatch_id]

        endpoint_health_events = []
        for endpoint_uri, http_request_completed_events in http_request_completed_events_per_endpoint.items():
            http_request_durations_ms = [e.duration_ms for e in http_request_completed_events] or [0]

            processed_event_count += len(http_request_completed_events)
            event = BatchScoreMinibatchEndpointHealthEvent(
                minibatch_id=minibatch_id,
                scoring_url=endpoint_uri,
                quota_audience=self._start_event_per_minibatch[minibatch_id].quota_audience,
                logging_metadata=logging_metadata,

                http_request_count=len(http_request_completed_events),
                http_request_succeeded_count=len(
                    [e for e in http_request_completed_events if self._is_request_succeeded(e)]),
                http_request_user_error_count=len(
                    [e for e in http_request_completed_events if self._is_request_user_error(e)]),
                http_request_system_error_count=len(
                    [e for e in http_request_completed_events if self._is_request_system_error(e)]),

                http_request_duration_p0_ms=float(np.min(http_request_durations_ms)),
                http_request_duration_p50_ms=float(np.percentile(http_request_durations_ms, 50)),
                http_request_duration_p90_ms=float(np.percentile(http_request_durations_ms, 90)),
                http_request_duration_p95_ms=float(np.percentile(http_request_durations_ms, 95)),
                http_request_duration_p99_ms=float(np.percentile(http_request_durations_ms, 99)),
                http_request_duration_p100_ms=float(np.max(http_request_durations_ms)),
            )
            endpoint_health_events.append(event)

        lu.get_logger().info(f"Minibatch {minibatch_id}: Successfully summarized "
                             f"http_request_completed_events: {processed_event_count}. ")

        return endpoint_health_events

    def _is_request_succeeded(self, event: BatchScoreEvent) -> bool:
        return event.response_code and 200 <= abs(event.response_code) < 300

    def _is_request_user_error(self, event: BatchScoreEvent) -> bool:
        return event.response_code and 400 <= abs(event.response_code) < 500

    def _is_request_system_error(self, event: BatchScoreEvent) -> bool:
        return event.response_code and 500 <= abs(event.response_code) < 600

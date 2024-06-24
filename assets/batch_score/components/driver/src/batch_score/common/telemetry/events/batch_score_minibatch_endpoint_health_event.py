# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains the definition for the batch score minibatch endpoint health event."""

from dataclasses import dataclass, field
from .batch_score_event import BatchScoreEvent


# TODO: Add comments to describe each field
@dataclass
class BatchScoreMinibatchEndpointHealthEvent(BatchScoreEvent):
    """Defines the batch score minibatch endpoint health event. Only used in BatchPool jobs."""

    @property
    def name(self):
        """Get the name of the event."""
        return "BatchScore.Minibatch.EndpointHealth"

    minibatch_id: str = field(init=True, default=None)
    scoring_url: str = field(init=True, default=None)
    batch_pool: str = field(init=True, default=None)
    quota_audience: str = field(init=True, default=None)

    # HTTP request counts

    """How many HTTP requests were made?"""
    http_request_count: int = field(init=True, default=None)

    """How many HTTP requests succeeded?"""
    http_request_succeeded_count: int = field(init=True, default=None)

    """How many HTTP requests failed due to a user error?"""
    http_request_user_error_count: int = field(init=True, default=None)

    """How many HTTP requests failed due to a system error?"""
    http_request_system_error_count: int = field(init=True, default=None)

    # HTTP request durations

    """How long does the shortest request take?"""
    http_request_duration_p0_ms: float = field(init=True, default=None)

    """How long does the average request take?"""
    http_request_duration_p50_ms: float = field(init=True, default=None)

    """How long does the 90th percentile request take?"""
    http_request_duration_p90_ms: float = field(init=True, default=None)

    """How long does the 95th percentile request take?"""
    http_request_duration_p95_ms: float = field(init=True, default=None)

    """How long does the 99th percentile request take?"""
    http_request_duration_p99_ms: float = field(init=True, default=None)

    """How long does the longest request take?"""
    http_request_duration_p100_ms: float = field(init=True, default=None)

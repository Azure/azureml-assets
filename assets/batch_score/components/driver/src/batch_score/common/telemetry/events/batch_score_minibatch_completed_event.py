# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains the definition for the batch score minibatch completed event."""

from dataclasses import dataclass, field
from .batch_score_event import BatchScoreEvent


# TODO: Add comments to describe each field
@dataclass
class BatchScoreMinibatchCompletedEvent(BatchScoreEvent):
    """Defines the batch score minibatch completed event."""

    @property
    def name(self):
        """Get the name of the event."""
        return "BatchScore.Minibatch.Completed"

    minibatch_id: str = field(init=True, default=None)
    scoring_url: str = field(init=True, default=None)
    batch_pool: str = field(init=True, default=None)
    quota_audience: str = field(init=True, default=None)

    # Tokens

    """What was the sum of the prompt tokens across all rows in the minibatch?"""
    total_prompt_tokens: int = field(init=True, default=None)

    """What was the sum of the completion tokens across all rows in the minibatch?"""
    total_completion_tokens: int = field(init=True, default=None)

    # Rows

    """How many rows were in the minibatch?"""
    input_row_count: int = field(init=True, default=None)

    """How many of the input rows were successfully processed?"""
    succeeded_row_count: int = field(init=True, default=None)

    """How many of the input rows failed to be processed?"""
    failed_row_count: int = field(init=True, default=None)

    """How many rows were saved as output?"""
    output_row_count: int = field(init=True, default=None)

    # HTTP request counts

    """How many HTTP requests were made?"""
    http_request_count: int = field(init=True, default=None)

    """How many HTTP requests succeeded?"""
    http_request_succeeded_count: int = field(init=True, default=None)

    """How many HTTP requests failed due to a user error?"""
    http_request_user_error_count: int = field(init=True, default=None)

    """How many HTTP requests failed due to a system error?"""
    http_request_system_error_count: int = field(init=True, default=None)

    """How many HTTP requests were retried?"""
    http_request_retry_count: int = field(init=True, default=None)

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

    # Progress durations
    """Note: Each completed row counts as progress regardless of whether the row succeeded or failed."""

    """How long after the start of the minibatch does the first row complete?"""
    progress_duration_p0_ms: float = field(init=True, default=None)

    """How long after the start of the minibatch are half of the rows complete?"""
    progress_duration_p50_ms: float = field(init=True, default=None)

    """How long after the start of the minibatch are 90% of the rows complete?"""
    progress_duration_p90_ms: float = field(init=True, default=None)

    """How long after the start of the minibatch are 95% of the rows complete?"""
    progress_duration_p95_ms: float = field(init=True, default=None)

    """How long after the start of the minibatch are 99% of the rows complete?"""
    progress_duration_p99_ms: float = field(init=True, default=None)

    """How long after the start of the minibatch does the final row complete?"""
    progress_duration_p100_ms: float = field(init=True, default=None)

    """ How long after the start of the minibatch does the minibatch complete?
        There may be a post-processing delay between the completion of the final row in the minibatch
        and the completion of the minibatch.
        In this case, total_duration_ms will be greater than progress_duration_p100_ms.
    """
    total_duration_ms: float = field(init=True, default=None)

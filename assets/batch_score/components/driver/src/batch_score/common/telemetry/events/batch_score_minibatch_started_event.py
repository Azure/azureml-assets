# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


"""This file contains the definition for the batch score minibatch started event."""

from dataclasses import dataclass, field
from .batch_score_event import BatchScoreEvent

from .. import logging_utils as lu


@dataclass
class BatchScoreMinibatchStartedEvent(BatchScoreEvent):
    """Defines the batch score minibatch started event."""

    @property
    def name(self):
        """Get the name of the event."""
        return "BatchScore.Minibatch.Started"

    minibatch_id: str = field(init=True, default_factory=lu.get_mini_batch_id)
    scoring_url: str = field(init=True, default=None)

    """How many rows were in the minibatch?"""
    input_row_count: int = field(init=True, default=None)

    """How many retries were executed on this minibatch prior to the current one."""
    retry_count: int = field(init=True, default=0)

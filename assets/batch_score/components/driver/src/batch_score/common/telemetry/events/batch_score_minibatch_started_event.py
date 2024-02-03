# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


"""This file contains the definition for the batch score minibatch started event."""

from dataclasses import dataclass, field
from .batch_score_event import BatchScoreEvent

from .. import logging_utils as lu


# TODO: Add comments to describe each field
@dataclass
class BatchScoreMinibatchStartedEvent(BatchScoreEvent):
    """Defines the batch score minibatch started event."""

    @property
    def name(self):
        """Get the name of the event."""
        return "BatchScore.Minibatch.Started"

    minibatch_id: str = field(init=True, default_factory=lu.get_mini_batch_id)
    scoring_url: str = field(init=True, default=None)
    batch_pool: str = field(init=True, default_factory=lu.get_batch_pool)
    quota_audience: str = field(init=True, default_factory=lu.get_quota_audience)

    # Number of rows in the minibatch
    input_row_count: int = field(init=True, default=None)

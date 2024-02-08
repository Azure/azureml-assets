# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains the definition for batch score request started event."""

from dataclasses import dataclass, field
from .batch_score_event import BatchScoreEvent


# TODO: Add comments to describe each field
@dataclass
class BatchScoreRequestStartedEvent(BatchScoreEvent):
    """Defines the batch score request started event."""

    @property
    def name(self):
        """Get the name of the event."""
        return "BatchScore.Request.Started"

    minibatch_id: str = field(init=True, default=None)
    input_row_id: str = field(init=True, default=None)
    worker_id: str = field(init=True, default=None)
    segmented_request_id: int = field(init=True, default=None)
    scoring_url: str = field(init=True, default=None)

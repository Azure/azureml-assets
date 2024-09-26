# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains the definition for batch score worker increased event."""

from dataclasses import dataclass, field
from .batch_score_event import BatchScoreEvent


# TODO: Add comments to describe each field
@dataclass
class BatchScoreWorkerIncreasedEvent(BatchScoreEvent):
    """Defines the batch score worker increased event."""

    @property
    def name(self):
        """Get the name of the event."""
        return "BatchScore.Worker.Increased"

    active_minibatch_count: int = field(init=True, default=0)
    previous_active_worker_count: int = field(init=True, default=0)
    current_active_worker_count: int = field(init=True, default=0)
    p90_wait_time_ms: float = field(init=True, default=0)

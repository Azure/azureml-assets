# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import dataclass, field
from .batch_score_event import BatchScoreEvent


# TODO: Add comments to describe each field
@dataclass
class BatchScoreWorkerDecreasedEvent(BatchScoreEvent):

    @property
    def name(self):
        return "BatchScore.Worker.Decreased"

    active_minibatch_count: int = field(init=True, default=0)
    previous_active_worker_count: int = field(init=True, default=0)
    current_active_worker_count: int = field(init=True, default=0)
    p90_wait_time_ms: float = field(init=True, default=0)

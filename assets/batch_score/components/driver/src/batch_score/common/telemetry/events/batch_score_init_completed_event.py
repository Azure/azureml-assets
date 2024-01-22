# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import dataclass, field
from .batch_score_event import BatchScoreEvent

# TODO: Add comments to describe each field
@dataclass
class BatchScoreInitCompletedEvent(BatchScoreEvent):

    @property
    def name(self):
        return "BatchScore.Init.Completed"

    init_duration_ms: float = field(init=True, default=0)
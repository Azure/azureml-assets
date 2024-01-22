# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import dataclass
from .batch_score_event import BatchScoreEvent


@dataclass
class BatchScoreInitStartedEvent(BatchScoreEvent):

    @property
    def name(self):
        return "BatchScore.Init.Started"

    pass

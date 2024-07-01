# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains the definition for the batch score init completed event."""

from dataclasses import dataclass, field
from .batch_score_event import BatchScoreEvent


# TODO: Add comments to describe each field
@dataclass
class BatchScoreInitCompletedEvent(BatchScoreEvent):
    """Defines the batch score init completed event."""

    @property
    def name(self):
        """Get the name of the event."""
        return "BatchScore.Init.Completed"

    init_duration_ms: float = field(init=True, default=0)

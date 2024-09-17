# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains the definition for the batch score init started event."""

from dataclasses import dataclass
from .batch_score_event import BatchScoreEvent


@dataclass
class BatchScoreInitStartedEvent(BatchScoreEvent):
    """Defines the batch score init started event."""

    @property
    def name(self):
        """Get the name of the event."""
        return "BatchScore.Init.Started"

    pass

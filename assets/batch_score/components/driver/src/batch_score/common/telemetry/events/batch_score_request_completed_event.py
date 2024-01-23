# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains the definition for the batch score request completed event."""

from dataclasses import dataclass, field
from .batch_score_event import BatchScoreEvent


# TODO: Add comments to describe each field
@dataclass
class BatchScoreRequestCompletedEvent(BatchScoreEvent):
    """Defines the batch score request completed event."""

    @property
    def name(self):
        """Get the name of the event."""
        return "BatchScore.Request.Completed"

    minibatch_id: str = field(init=True, default=None)
    input_row_id: str = field(init=True, default=None)
    x_ms_client_request_id: str = field(init=True, default=None)
    worker_id: str = field(init=True, default=None)
    segmented_request_id: int = field(init=True, default=None)
    scoring_url: str = field(init=True, default=None)
    is_successful: bool = field(init=True, default=None)
    is_retriable: bool = field(init=True, default=None)
    response_code: int = field(init=True, default=None)
    model_response_code: int = field(init=True, default=None)
    prompt_tokens: int = field(init=True, default=None)
    completion_tokens: int = field(init=True, default=None)
    duration_ms: float = field(init=True, default=None)

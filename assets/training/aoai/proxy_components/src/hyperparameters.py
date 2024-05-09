# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Hyperparameters for Azure Open AI Finetuning."""
from typing import Dict, Optional
from pydantic import BaseModel, Field


class Hyperparameters(BaseModel):
    """Hyperparameters for finetuning."""

    n_epochs: Optional[int] = Field(...)
    batch_size: Optional[int] = Field(...)
    learning_rate_multiplier: Optional[float] = Field(...)

    def get_dict(self) -> Dict[str, str]:
        """Get dictionary of hyperparameters."""
        return {key: str(value) for key, value in self.dict().items() if value is not None}


class Hyperparameters_1P(BaseModel):
    """Hyperparameters available for 1P customers for finetuning."""

    n_ctx: Optional[int] = Field(...)
    lora_dim: Optional[int] = Field(...)
    weight_decay_multiplier: Optional[float] = Field(...)

    def get_dict(self) -> Dict[str, str]:
        """Get dictionary of hyperparameters."""
        return {key: str(value) for key, value in self.dict().items() if value is not None}

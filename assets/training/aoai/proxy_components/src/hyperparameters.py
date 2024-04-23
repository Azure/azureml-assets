# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Hyperparameters for Azure Open AI Finetuning."""
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class Hyperparameters(BaseModel):
    """Hyperparameters for finetuning."""

    n_epochs: Optional[int] = Field(...)
    batch_size: Optional[int] = Field(...)
    learning_rate_multiplier: Optional[float] = Field(...)

    def get_dict(self, exclude_none=True) -> Dict[str, Any]:
        """Dictionary of hyperparameters."""
        if exclude_none is False:
            return self.dict()
        return {key: value for key, value in self.dict().items() if value is not None}


class Hyperparameters_1P(BaseModel):
    """Hyperparameters available for 1P customers for finetuning."""

    export_merged_weights: Optional[bool] = Field(...)
    completion_override: Optional[bool] = Field(...)
    full_finetune: Optional[bool] = Field(...)
    lora_v2: Optional[bool] = Field(...)
    lora_dimensions: Optional[int] = Field(...)
    context_window: Optional[int] = Field(...)
    file_spm_rate: Optional[float] = Field(..., ge=0, le=1)
    weight_decay_multiplier: Optional[float] = Field(...)
    prompt_loss_weight: Optional[float] = Field(...)
    trim_mode: Optional[str] = Field(...)
    check_point_interval: Optional[int] = Field(...)
    num_steps: Optional[int] = Field(...)
    shuffle_type: Optional[str] = Field(...)

    def get_dict(self, exclude_none=True) -> Dict[str, Any]:
        """Dictionary of hyperparameters."""
        if exclude_none is False:
            return self.dict()
        return {key: value for key, value in self.dict().items() if value is not None}
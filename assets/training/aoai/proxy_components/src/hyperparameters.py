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

    def get_dict(self) -> Dict[str, str]:
        """Dictionary of hyperparameters."""
        return {key: str(value) for key, value in self.dict().items() if value is not None}


class Hyperparameters_1P(BaseModel):
    """Hyperparameters available for 1P customers for finetuning."""

    ExportMergedWeights: Optional[bool] = Field(...)
    CompletionOverride: Optional[bool] = Field(...)
    FullFineTune: Optional[bool] = Field(...)
    LoraV2: Optional[bool] = Field(...)
    LoraDimensions: Optional[int] = Field(...)
    ContextWindow: Optional[int] = Field(...)
    FileSPMRate: Optional[float] = Field(..., ge=0, le=1)
    WeightDecayMultiplier: Optional[float] = Field(...)
    PromptLossWeight : Optional[float] = Field(...)
    TrimMode: Optional[str] = Field(...)
    CheckPointInterval: Optional[int] = Field(...)
    NumSteps: Optional[int] = Field(...)
    ShuffleType: Optional[str] = Field(...)

    def get_dict(self) -> Dict[str, str]:
        """Dictionary of hyperparameters."""
        return {key: str(value) for key, value in self.dict().items() if value is not None}

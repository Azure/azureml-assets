# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Signal type enum."""

from types import DynamicClassAttribute
from enum import Enum


class SignalType(Enum):
    """Defines the type of signal."""

    DATA_DRIFT = 1
    PREDICTION_DRIFT = 2
    DATA_QUALITY = 3
    FEATURE_ATTRIBUTION_DRIFT = 4
    CUSTOM_SIGNAL = 5
    GENERATION_SAFETY_SIGNAL_QUALITY = 6

    @DynamicClassAttribute
    def name(self):
        """Return the name of the enum value."""
        name = super(SignalType, self).name

        if name == "DATA_DRIFT":
            return "DataDrift"
        elif name == "PREDICTION_DRIFT":
            return "PredictionDrift"
        elif name == "DATA_QUALITY":
            return "DataQuality"
        elif name == "FEATURE_ATTRIBUTION_DRIFT":
            return "FeatureAttributionDrift"
        elif name == "CUSTOM_SIGNAL":
            return "Custom"
        elif name == "GENERATION_SAFETY_SIGNAL_QUALITY":
            return "GenerationSafetyQuality"

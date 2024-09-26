# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Data augmentation constants."""

# COMPONENT META
COMPONENT_NAME = "oss_augment_data"

# DATA AUGMENTATION FILE FORMAT
SUPPORTED_FILE_FORMATS = [".jsonl"]

DEFAULT_SEED = 1000
DEFAULT_PROPORTION = 1.0

class TelemetryConstants:
    """Telemetry constants that describe various activities performed by the distillation components."""
    DATA_AUGMENTATOR = "augment_data"

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Feature Selector type enum."""

from types import DynamicClassAttribute
from enum import Enum


class FeatureSelectorType(Enum):
    """Defines the type of signal."""

    ALL = 1
    SUBSET = 2
    TOP_N_BY_ATTRIBUTION = 3

    @DynamicClassAttribute
    def name(self):
        """Return the name of the enum value."""
        name = super(FeatureSelectorType, self).name

        if name == "ALL":
            return "All"
        elif name == "SUBSET":
            return "Subset"
        elif name == "TOP_N_BY_ATTRIBUTION":
            return "TopNByAttribution"

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Factory class to generate AppInsight object."""

from appinsights_recorder import AppInsightsRecorder


def get_recorders():
    """Get appinsight object."""
    return AppInsightsRecorder()

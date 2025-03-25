# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Factory class to generate AppInsight object."""

from appinsights_recorder import AppInsightsRecorder


def get_recorders():
    """Method to get appinsight object."""
    return AppInsightsRecorder()

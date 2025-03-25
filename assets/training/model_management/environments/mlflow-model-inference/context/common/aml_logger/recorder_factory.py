# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from appinsights_recorder import AppInsightsRecorder


def get_recorders():
    return AppInsightsRecorder()

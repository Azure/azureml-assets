# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from appinsights_recorder import AppInsightsRecorder


def get_recorders():
    return AppInsightsRecorder()

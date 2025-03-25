# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import os
from appinsights_recorder import AppInsightsRecorder


def get_recorders():
    return AppInsightsRecorder()

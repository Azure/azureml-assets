import os
from appinsights_recorder import AppInsightsRecorder


def get_recorders():
    return AppInsightsRecorder()

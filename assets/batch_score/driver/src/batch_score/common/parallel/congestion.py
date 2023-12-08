# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Congestion detector."""

import os
from abc import ABC, abstractmethod
from enum import Enum

import pandas as pd

from ..configuration.client_settings import ClientSettingsKey, ClientSettingsProvider
from ..scoring.scoring_result import ScoringResultStatus
from ..telemetry import logging_utils as lu
from .request_metrics import RequestMetrics


class CongestionState(Enum):
    """Congestion state."""

    CONGESTED = 1
    SATURATED = 2
    FREE = 3
    UNKNOWN = 4


class CongestionDetector(ABC):
    """Congestion detector."""

    @abstractmethod
    def detect(self,
               request_metrics: RequestMetrics,
               start_time: pd.Timestamp,
               end_time: pd.Timestamp = None) -> CongestionState:
        """Detect congestion."""
        pass


class WaitTimeCongestionDetector(CongestionDetector):
    """Wait time congestion detector."""

    DEFAULT_CONGESTION_P90_THRESHOLD = 10
    DEFAULT_SATURATION_P90_THRESHOLD = 5

    def __init__(self, client_settings_provider: ClientSettingsProvider):
        """Init function."""
        self.__client_settings_provider = client_settings_provider
        self._refresh_thresholds()

    def detect(self,
               request_metrics: RequestMetrics,
               start_time: pd.Timestamp,
               end_time: pd.Timestamp = None) -> CongestionState:
        """Detect congestion from request total wait time metrics."""
        self._refresh_thresholds()

        metrics = request_metrics.get_metrics(start_time, end_time)

        # By design, only succeeded responses should contribute to an increase or decrease worker concurrency.
        # Today, the column metrics[RequestMetrics.COLUMN_RESPONSE_CODE] can contain one of three values:
        #  1. ScoringResultStatus.SUCCESS
        #  2. ScoringResultStatus.FAILURE
        #  3. an integer representing a retriable HTTP response code
        # So for now, we keep only the rows where the response code == ScoringResultStatus.SUCCESS.
        metrics = metrics[metrics[RequestMetrics.COLUMN_RESPONSE_CODE] == ScoringResultStatus.SUCCESS]

        # Long term, we should simplify the column RequestMetrics.COLUMN_RESPONSE_CODE
        # to have only enum values or only integer status codes.
        # TODO: https://dev.azure.com/msdata/Vienna/_workitems/edit/2832428

        response_count = len(metrics.index)

        p90_wait_time: int = 0
        request_count: int = 0
        result: CongestionState

        if response_count == 0:
            result = CongestionState.UNKNOWN
        else:
            df = metrics.groupby(by=RequestMetrics.COLUMN_REQUEST_ID, sort=False) \
                     .max(RequestMetrics.COLUMN_REQUEST_TOTAL_WAIT_TIME)
            p90_wait_time = df[RequestMetrics.COLUMN_REQUEST_TOTAL_WAIT_TIME].quantile(0.9, interpolation="linear")
            request_count = len(df.index)

            if p90_wait_time < self.__saturation_threshold:
                result = CongestionState.FREE
            elif p90_wait_time >= self.__congestion_threshold:
                result = CongestionState.CONGESTED
            else:
                result = CongestionState.SATURATED

        msg = "WaitTimeCongestionDetector response_count: {} request_count: {}, p90_wait_time: {} result: {}".format(
                response_count, request_count, p90_wait_time, result)
        lu.get_logger().info(msg)

        return result

    def _refresh_thresholds(self):
        self.__congestion_threshold = float(
            os.environ.get("BATCH_SCORE_CONGESTION_THRESHOLD_P90_WAITTIME")
            or self.__client_settings_provider.get_client_setting(ClientSettingsKey.CONGESTION_THRESHOLD_P90_WAIT_TIME)
            or self.DEFAULT_CONGESTION_P90_THRESHOLD)

        self.__saturation_threshold = float(
            os.environ.get("BATCH_SCORE_SATURATION_THRESHOLD_P90_WAITTIME")
            or self.__client_settings_provider.get_client_setting(ClientSettingsKey.SATURATION_THRESHOLD_P90_WAIT_TIME)
            or self.DEFAULT_SATURATION_P90_THRESHOLD)

        msg = "WaitTimeCongestionDetector using congestion threshold: {}, saturation threshold: {}".format(
                self.__congestion_threshold, self.__saturation_threshold)
        lu.get_logger().info(msg)

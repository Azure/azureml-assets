# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""The class for congestion."""

import os
import pandas as pd
from abc import ABC, abstractmethod
from enum import Enum
from .request_metrics import RequestMetrics
from ..utils.common.common import str2bool
from ..utils.scoring_utils import is_retriable
from ..utils import logging_utils as lu


class CongestionState(Enum):
    """Enum for congestion state."""

    CONGESTED = 1
    SATURATED = 2
    FREE = 3
    UNKNOWN = 4


class CongestionDetector(ABC):
    """Base class for congestion detector."""

    @abstractmethod
    def detect(
            self,
            request_metrics: RequestMetrics, start_time: pd.Timestamp, end_time: pd.Timestamp = None
    ) -> CongestionState:
        """Detect interface."""
        pass


class WaitTimeCongestionDetector(CongestionDetector):
    """Wait time congestion detector."""

    DEFAULT_CONGESTION_P90_THRESHOLD = 10
    DEFAULT_SATURATION_P90_THRESHOLD = 5
    DEFAULT_USE_TOTAL_WAIT_TIME = True

    def __init__(self):
        """Init method."""
        congestion_override = os.environ.get("BATCH_SCORE_CONGESTION_THRESHOLD_P90_WAITTIME")
        self.__congestion_threshold = (
            float(congestion_override) if congestion_override else self.DEFAULT_CONGESTION_P90_THRESHOLD)

        saturation_override = os.environ.get("BATCH_SCORE_SATURATION_THRESHOLD_P90_WAITTIME")
        self.__saturation_threshold = (
            float(saturation_override) if saturation_override else self.DEFAULT_SATURATION_P90_THRESHOLD)

        total_wait_time_override = os.environ.get("BATCH_SCORE_CONGESTION_USE_TOTAL_WAIT_TIME")
        self.__use_request_total_wait_time = (
            str2bool(total_wait_time_override) if total_wait_time_override else self.DEFAULT_USE_TOTAL_WAIT_TIME)

        lu.get_logger().info(
            f"WaitTimeCongestionDetector using congestion threshold: {self.__congestion_threshold}, "
            f"saturation threshold: {self.__saturation_threshold}, "
            f"use total wait time: {self.__use_request_total_wait_time}")

    def detect(
            self,
            request_metrics: RequestMetrics,
            start_time: pd.Timestamp,
            end_time: pd.Timestamp = None
    ) -> CongestionState:
        """Detect method."""
        requests = request_metrics.get_metrics(start_time, end_time)
        response_count = len(requests.index)

        p90_wait_time: int = 0
        request_count: int = 0
        result: CongestionState

        if response_count == 0:
            result = CongestionState.UNKNOWN
        else:
            if self.__use_request_total_wait_time:
                df = requests.groupby(
                    by=RequestMetrics.COLUMN_REQUEST_ID, sort=False).max(
                        RequestMetrics.COLUMN_REQUEST_TOTAL_WAIT_TIME)
                p90_wait_time = df[RequestMetrics.COLUMN_REQUEST_TOTAL_WAIT_TIME].quantile(
                    0.9, interpolation="linear")
                request_count = len(df.index)
            else:
                df = requests.groupby(
                    by=RequestMetrics.COLUMN_REQUEST_ID, sort=False).sum(
                        RequestMetrics.COLUMN_ADDITIONAL_WAIT_TIME)
                p90_wait_time = df[RequestMetrics.COLUMN_ADDITIONAL_WAIT_TIME].quantile(
                    0.9, interpolation="linear")
                request_count = len(df.index)

            if p90_wait_time < self.__saturation_threshold:
                result = CongestionState.FREE
            elif p90_wait_time >= self.__congestion_threshold:
                result = CongestionState.CONGESTED
            else:
                result = CongestionState.SATURATED

        lu.get_logger().info(
            f"WaitTimeCongestionDetector response_count: {response_count} request_count: {request_count},"
            f" p90_wait_time: {p90_wait_time} result: {result}")

        return result


class ThrottleCountCongestionDetector(CongestionDetector):
    """Throttle count congestion detector."""

    DEFAULT_CONGESTION_THRESHOLD = 0.15
    DEFAULT_SATURATION_THRESHOLD = 0.05

    def __init__(self):
        """Init method."""
        congestion_override = os.environ.get("BATCH_SCORE_CONGESTION_THRESHOLD_THROTTLE_PERCENTAGE")
        self.__congestion_threshold = (
            float(congestion_override) if congestion_override else self.DEFAULT_CONGESTION_THRESHOLD)

        saturation_override = os.environ.get("BATCH_SCORE_SATURATION_THRESHOLD_THROTTLE_PERCENTAGE")
        self.__saturation_threshold = (
            float(saturation_override) if saturation_override else self.DEFAULT_SATURATION_THRESHOLD)

        lu.get_logger().info(
            f"ThrottleCountCongestionDetector using congestion threshold: {self.__congestion_threshold},"
            f" saturation threshold: {self.__saturation_threshold}")

    def detect(
            self,
            request_metrics: RequestMetrics, start_time: pd.Timestamp, end_time: pd.Timestamp = None
    ) -> CongestionState:
        """Detect method."""
        requests = request_metrics.get_metrics(start_time, end_time)
        response_count = len(requests.index)
        df = requests

        df = df[df.apply(
            lambda row: is_retriable(
                response_status=row[RequestMetrics.COLUMN_RESPONSE_CODE],
                response_payload=row[RequestMetrics.COLUMN_RESPONSE_PAYLOAD],
                model_response_code=row[RequestMetrics.COLUMN_MODEL_RESPONSE_CODE],
                model_response_reason=row[RequestMetrics.COLUMN_MODEL_RESPONSE_REASON]),
            axis=1)]

        retry_count = len(df.index)

        result: CongestionState

        if response_count == 0:
            result = CongestionState.UNKNOWN
        else:
            retry_rate = retry_count / response_count

            if retry_rate < self.__saturation_threshold:
                result = CongestionState.FREE
            elif retry_rate >= self.__congestion_threshold:
                result = CongestionState.CONGESTED
            else:
                result = CongestionState.SATURATED

        lu.get_logger().info(
            f"ThrottleCountCongestionDetector response_count: {response_count} retry_count: {retry_count}"
            f" result: {result}")

        return result

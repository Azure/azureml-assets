# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""The class for concurrency adjustment."""

import os
import pandas as pd
from . import congestion
from abc import ABC, abstractmethod
from .request_metrics import RequestMetrics
from ..utils import logging_utils as lu


class _ConcurrencyAdjustment:
    def __init__(self, new_concurrency: int, next_adjustment_delay: int):
        """Init method."""
        self.new_concurrency = new_concurrency
        self.next_adjustment_delay = next_adjustment_delay


# Abstract class for concurrency adjustment strategies
class _ConcurrencyAdjustmentStrategy(ABC):
    @abstractmethod
    def calculate_next_concurrency(self, current_concurrency: int) -> _ConcurrencyAdjustment:
        """Calculate the next concurrency."""
        pass


# Additive increase/multiplicative decrease
# https://en.wikipedia.org/wiki/Additive_increase/multiplicative_decrease
class AIMD(_ConcurrencyAdjustmentStrategy):
    """Class for additive increase multiplicative decrease."""

    DEFAULT_ADJUSTMENT_INTERVAL = 180
    DEFAULT_ADDITIVE_INCREASE = 1
    DEFAULT_MULTIPLICATIVE_DECREASE = 0.75

    def __init__(self, request_metrics: RequestMetrics):
        """Init method."""
        self.__confidence: int = 0
        self._last_adjustment_time: pd.Timestamp = None
        self.__request_metrics = request_metrics

        congestion_detector_override = os.environ.get("BATCH_SCORE_CONGESTION_DETECTOR", "WaitTime")
        if congestion_detector_override == "WaitTime":
            self.__congestion_detector = congestion.WaitTimeCongestionDetector()
        else:
            self.__congestion_detector = congestion.ThrottleCountCongestionDetector()

        adjustment_interval_override = os.environ.get("BATCH_SCORE_CONCURRENCY_ADJUSTMENT_INTERVAL")
        self.__adjustment_interval = (
            int(adjustment_interval_override) if adjustment_interval_override else self.DEFAULT_ADJUSTMENT_INTERVAL
        )

        increase_override = os.environ.get("BATCH_SCORE_CONCURRENCY_INCREASE_AMOUNT")
        self.__additive_increase = (
            int(increase_override) if increase_override else self.DEFAULT_ADDITIVE_INCREASE)

        decrease_override = os.environ.get("BATCH_SCORE_CONCURRENCY_DECREASE_RATE")
        self.__multiplicative_decrease = (
            float(decrease_override) if decrease_override else self.DEFAULT_MULTIPLICATIVE_DECREASE)

        lu.get_logger().info(
            "AIMD: using configurations CongestionDetector: {}, adjustment_interval: {}, "
            "additive_increase: {}, multiplicative_decrease: {}".format(
                congestion_detector_override,
                self.__adjustment_interval,
                self.__additive_increase,
                self.__multiplicative_decrease))

    def calculate_next_concurrency(self, current_concurrency: int) -> _ConcurrencyAdjustment:
        """Calculate the next concurrency."""
        new_concurrency = current_concurrency
        now = pd.Timestamp.utcnow()

        congestion_state = self.__congestion_detector.detect(
            self.__request_metrics, self._last_adjustment_time, now)

        if congestion_state == congestion.CongestionState.FREE:
            self.__confidence = max(self.__confidence, 0)  # Reset confidence
            self.__confidence = min(self.__confidence + 1, 3)  # Increase confidence
            # Augment based on confidence
            new_concurrency = current_concurrency + self.__additive_increase + self.__confidence
        elif congestion_state == congestion.CongestionState.CONGESTED:
            self.__confidence = min(self.__confidence, 0)  # Reset confidence
            self.__confidence = max(self.__confidence - 1, -3)  # Decrease confidence
            # Minimum value concurrency is 1
            new_concurrency = max(int(current_concurrency * self.__multiplicative_decrease), 1)
        else:
            self.__confidence = 0  # Reset confidence

        lu.get_logger().info(
            "AIMD: current_concurrency: {} -- new_concurrency: {} -- __confidence: {}".format(
                current_concurrency,
                new_concurrency,
                self.__confidence))

        self._last_adjustment_time = now

        return _ConcurrencyAdjustment(new_concurrency, self.__adjustment_interval)

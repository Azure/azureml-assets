import os
from abc import ABC, abstractmethod

import pandas as pd

from ..configuration.client_settings import ClientSettingsKey, ClientSettingsProvider
from ..telemetry import logging_utils as lu
from . import congestion
from .request_metrics import RequestMetrics


class ConcurrencyAdjustment:
    def __init__(self, new_concurrency: int, next_adjustment_delay: int):
        self.new_concurrency = new_concurrency
        self.next_adjustment_delay = next_adjustment_delay

# Abstract class for concurrency adjustment strategies
class ConcurrencyAdjustmentStrategy(ABC):
    @abstractmethod
    def calculate_next_concurrency(self, current_concurrency: int) -> ConcurrencyAdjustment:
        pass

# Additive increase/multiplicative decrease
# https://en.wikipedia.org/wiki/Additive_increase/multiplicative_decrease
class AIMD(ConcurrencyAdjustmentStrategy):
    DEFAULT_ADJUSTMENT_INTERVAL = 180
    DEFAULT_ADDITIVE_INCREASE = 1
    DEFAULT_MULTIPLICATIVE_DECREASE = 0.75

    def __init__(
        self,
        request_metrics: RequestMetrics,
        client_settings_provider: ClientSettingsProvider,
    ):
        self.__confidence: int = 0
        self._last_adjustment_time: pd.Timestamp = None
        self.__request_metrics=request_metrics
        self.__congestion_detector = congestion.WaitTimeCongestionDetector(client_settings_provider)

        self.__client_settings_provider = client_settings_provider
        self._refresh_concurrency_settings()

    def calculate_next_concurrency(self, current_concurrency: int) -> ConcurrencyAdjustment:
        self._refresh_concurrency_settings()

        new_concurrency = current_concurrency
        now = pd.Timestamp.utcnow()

        congestionState = self.__congestion_detector.detect(self.__request_metrics, self._last_adjustment_time, now)

        if congestionState == congestion.CongestionState.FREE:
            self.__confidence = max(self.__confidence, 0) # Reset confidence
            self.__confidence = min(self.__confidence + 1, 3) # Increase confidence

            new_concurrency = current_concurrency + self.__additive_increase + self.__confidence # Augment based on confidence
        elif congestionState == congestion.CongestionState.CONGESTED:
            self.__confidence = min(self.__confidence, 0) # Reset confidence
            self.__confidence = max(self.__confidence - 1, -3) # Decrease confidence

            new_concurrency = max(int(current_concurrency * self.__multiplicative_decrease), 1) # Minimum value concurrency is 1
        else:
            self.__confidence = 0 # Reset confidence

        lu.get_logger().info("AIMD: current_concurrency: {} -- new_concurrency: {} -- __confidence: {}".format(
            current_concurrency,
            new_concurrency,
            self.__confidence))

        self._last_adjustment_time = now

        return ConcurrencyAdjustment(new_concurrency, self.__adjustment_interval)

    def _refresh_concurrency_settings(self):
        self.__adjustment_interval = int(float(
            os.environ.get("BATCH_SCORE_CONCURRENCY_ADJUSTMENT_INTERVAL")
            or self.__client_settings_provider.get_client_setting(ClientSettingsKey.CONCURRENCY_ADJUSTMENT_INTERVAL)
            or self.DEFAULT_ADJUSTMENT_INTERVAL))
        
        self.__additive_increase = int(float(
            os.environ.get("BATCH_SCORE_CONCURRENCY_INCREASE_AMOUNT")
            or self.__client_settings_provider.get_client_setting(ClientSettingsKey.CONCURRENCY_ADDITIVE_INCREASE)
            or self.DEFAULT_ADDITIVE_INCREASE))

        self.__multiplicative_decrease = float(
            os.environ.get("BATCH_SCORE_CONCURRENCY_DECREASE_RATE")
            or self.__client_settings_provider.get_client_setting(ClientSettingsKey.CONCURRENCY_MULTIPLICATIVE_DECREASE)
            or self.DEFAULT_MULTIPLICATIVE_DECREASE)

        lu.get_logger().info("AIMD: using configurations CongestionDetector: WaitTimeCongestionDetector, adjustment_interval: {}, additive_increase: {}, multiplicative_decrease: {}".format(
            self.__adjustment_interval,
            self.__additive_increase,
            self.__multiplicative_decrease))

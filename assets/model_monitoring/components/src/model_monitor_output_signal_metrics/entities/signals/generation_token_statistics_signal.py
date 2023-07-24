# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Builder class which creates a Generation Token Statistics signal output."""

from typing import List
from pyspark.sql import Row
from model_monitor_output_signal_metrics.entities.signal_type import SignalType
from model_monitor_output_signal_metrics.entities.signals.signal import Signal


class GenerationTokenStatisticsSignal(Signal):
    """Builder class which creates a Generation Token Statistics signal output."""

    def __init__(
        self,
        monitor_name: str,
        signal_name: str,
        metrics: List[Row],
    ):
        """Build a Generation Token Statistics signal."""
        super().__init__(
            monitor_name, signal_name, "1.0.0", SignalType.GENERATION_TOKEN_STATISTICS, metrics
        )

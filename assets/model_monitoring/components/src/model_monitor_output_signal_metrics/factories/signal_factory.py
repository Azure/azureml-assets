# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Builder class which creates a Signal output."""

from typing import List
from pyspark.sql import Row
from model_monitor_output_signal_metrics.entities.signal_type import SignalType
from model_monitor_output_signal_metrics.entities.signals.signal import Signal
from model_monitor_output_signal_metrics.entities.signals.generation_token_statistics_signal import (
    GenerationTokenStatisticsSignal,
)


class SignalFactory:
    """Builder class which creates a Signal output."""

    def produce(
        self,
        signal_type: SignalType,
        monitor_name: str,
        signal_name: str,
        metrics: List[Row],
    ) -> Signal:
        """Produce a signal of the given type."""
        if signal_type == SignalType.GENERATION_TOKEN_STATISTICS_SIGNAL.name:
            return GenerationTokenStatisticsSignal(
                monitor_name=monitor_name, signal_name=signal_name, metrics=metrics
            )
        else:
            raise Exception(
                f"Invalid signal type '{signal_type}'. Available signals are [{SignalType.GENERATION_TOKEN_STATISTICS_SIGNAL.name}]"
            )

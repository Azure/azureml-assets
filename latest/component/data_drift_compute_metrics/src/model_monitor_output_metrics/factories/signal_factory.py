# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Builder class which creates a Data Drift signal output."""

from typing import List
from pyspark.sql import Row
from model_monitor_output_metrics.entities.signal_type import SignalType
from model_monitor_output_metrics.entities.signals.signal import Signal
from model_monitor_output_metrics.entities.signals.data_drift_signal import (
    DataDriftSignal,
)
from model_monitor_output_metrics.entities.signals.prediction_drift_signal import (
    PredictionDriftSignal,
)
from model_monitor_output_metrics.entities.signals.data_quality_signal import (
    DataQualitySignal,
)
from model_monitor_output_metrics.entities.signals.feature_attribution_drift_signal import (
    FeatureAttributionDriftSignal,
)


class SignalFactory:
    """Builder class which creates a Signal output."""

    def produce(
        self,
        signal_type: SignalType,
        monitor_name: str,
        signal_name: str,
        metrics: List[Row],
        baseline_histogram: List[Row] = None,
        target_histogram: List[Row] = None,
    ) -> Signal:
        """Produce a signal of the given type."""
        if signal_type == SignalType.DATA_DRIFT.name:
            return DataDriftSignal(
                monitor_name=monitor_name,
                signal_name=signal_name,
                metrics=metrics,
                baseline_histogram=baseline_histogram,
                target_histogram=target_histogram,
            )
        elif signal_type == SignalType.PREDICTION_DRIFT.name:
            return PredictionDriftSignal(
                monitor_name=monitor_name, signal_name=signal_name, metrics=metrics
            )
        elif signal_type == SignalType.DATA_QUALITY.name:
            return DataQualitySignal(
                monitor_name=monitor_name, signal_name=signal_name, metrics=metrics
            )
        elif signal_type == SignalType.FEATURE_ATTRIBUTION_DRIFT.name:
            return FeatureAttributionDriftSignal(
                monitor_name=monitor_name, signal_name=signal_name, metrics=metrics
            )
        else:
            raise Exception(
                f"Invalid signal type '{signal_type}'. Available signals are [{SignalType.DATA_DRIFT.name}, {SignalType.PREDICTION_DRIFT.name}, {SignalType.DATA_QUALITY.name}, {SignalType.FEATURE_ATTRIBUTION_DRIFT.name}]" # noqa
            )

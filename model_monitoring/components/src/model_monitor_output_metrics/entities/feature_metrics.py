# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Feature metric class."""

from typing import List


class FeatureMetrics:
    """A class which holds feature-specific metrics of a given signal."""

    def __init__(
        self,
        data_type: str,
        feature_name: str,
        metrics: List[dict],
        run_metrics,
        histogram: str,
    ):
        """Construct a FeatureMetric instance."""
        self.data_type = data_type
        self.metrics = metrics
        self.feature_name = feature_name
        self.run_metrics = run_metrics
        self.histogram = histogram

    def to_dict(self) -> dict:
        """Convert the class into a dictionary."""
        metrics = {"dataType": self.data_type}

        if self.histogram is not None:
            metrics["histogram"] = self.histogram

        if self.metrics is not None:
            metrics["metrics"] = self.metrics

        if self.run_metrics is not None:
            metrics["runMetrics"] = self.run_metrics
        return metrics

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Row count metric class."""
from typing import List


class RowCountMetrics:
    """A class which holds the row count metrics of a signal."""

    def __init__(self, metrics: List[dict]):
        """Construct a RowCountMetrics instance."""
        self.baseline_count = None
        self.target_count = None
        for metric in metrics:
            if metric["metric_name"] == "BaselineRowCount":
                self.baseline_count = metric["metric_value"]
            if metric["metric_name"] == "TargetRowCount":
                self.target_count = metric["metric_value"]

    def has_value(self) -> bool:
        """Check if the row count metrics have values."""
        return self.baseline_count is not None or self.target_count is not None

    def to_dict(self) -> dict:
        """Convert the class into a dictionary."""
        metrics = {}
        if self.baseline_count is not None:
            metrics["baselineCount"] = self.baseline_count
        if self.target_count is not None:
            metrics["targetCount"] = self.target_count
        return metrics

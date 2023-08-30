# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Builder class which creates a metrics object."""

from typing import List
import uuid


class MockRunMetricClient:
    """Mock client leveraged to retrieve and publish run metrics."""

    def __init__(self):
        """Construct a MetricOutputBuilder instance."""
        self.run_id = str(uuid.uuid4())

    def get_or_create_run_id(
        self, monitor_name: str, signal_name: str, metric_name: str, groups: List[str]
    ) -> str:
        """Get or create a run id for a given monitor, signal, metric and groups."""
        return self.run_id

    def publish_metric(self, run_id: str, value: float, threshold, step: int):
        """Publish a metric to the run metrics store."""
        print(f"Publishing metric to run id '{run_id}'.")

    def publish_metrics(self, run_id: str, metrics: dict, step: int):
        """Publish metrics to the run metrics store."""
        print(f"Publishing metrics to run id '{run_id}'.")

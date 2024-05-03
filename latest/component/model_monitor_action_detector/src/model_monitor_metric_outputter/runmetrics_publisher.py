# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Client leveraged to publish signal metrics to runs."""

from runmetric_client import RunMetricClient
from shared_utilities.constants import (
    GROUPS,
    THRESHOLD,
    VALUE,
    TIMESERIES,
    TIMESERIES_RUN_ID,
    TIMESERIES_METRIC_NAMES,
    TIMESERIES_METRIC_NAMES_VALUE,
    TIMESERIES_METRIC_NAMES_THRESHOLD,
)


class RunMetricPublisher:
    """Client leveraged to publish signal metrics to runs."""

    def __init__(self, runmetric_client: RunMetricClient):
        """Construct a MetricOutputBuilder instance."""
        self.runmetric_client: RunMetricClient = runmetric_client

    def publish_metrics(self, metrics: dict, metric_step: int):
        """Publish metrics to AML Run Metrics."""
        for metric_name in metrics:
            print(f"Publishing metrics for '{metric_name}'.")
            self._publish_metrics(metrics[metric_name], metric_step)

    def _publish_metrics(self, metrics: dict, metric_step: int):
        """Publish signal metrics to run."""
        # check if run metrics is present at this level
        if TIMESERIES in metrics:
            self._publish_metric_to_run(metrics, metric_step)

        # Iterate through nested groups and call publish metrics recursively
        if GROUPS in metrics:
            for group in metrics[GROUPS]:
                self._publish_metrics(metrics[GROUPS][group], metric_step)

    def _publish_metric_to_run(self, metrics: dict, metric_step: int):

        metrics_to_publish = {}
        timeseries = metrics[TIMESERIES]
        metric_names = timeseries[TIMESERIES_METRIC_NAMES]

        if VALUE in metrics and metrics[VALUE] is not None and metrics[VALUE] != "":
            metrics_to_publish[metric_names[TIMESERIES_METRIC_NAMES_VALUE]] = float(
                metrics[VALUE]
            )

        if THRESHOLD in metrics and metrics[THRESHOLD] is not None and metrics[THRESHOLD] != "":
            metrics_to_publish[metric_names[TIMESERIES_METRIC_NAMES_THRESHOLD]] = float(
                metrics[THRESHOLD]
            )

        run_id = timeseries[TIMESERIES_RUN_ID]

        self.runmetric_client.publish_metrics(run_id, metrics_to_publish, metric_step)

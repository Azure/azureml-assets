# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Builder class which creates a metrics object."""

import datetime
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
    """Builder class which creates a metrics object."""

    def __init__(self, runmetric_client : RunMetricClient):
        """Construct a MetricOutputBuilder instance."""
        self.runmetric_client : RunMetricClient = runmetric_client

    def publish_metrics(self, metrics: dict, metric_step: int):
        """Publish metrics to AML Run Metrics."""

        # check if run metrics is present at this level
        if(TIMESERIES in metrics):
            print("Publishing metrics.")
            self._publish_metric_to_run(metrics)

        # Iterate through nested groups and call publish metrics recursively
        if(GROUPS in metrics):
            for group in metrics[GROUPS]:
                self.publish_metrics(metrics[group], metric_step)

    def _publish_metric_to_run(self, metrics: dict, metric_step: int):

        metrics_to_publish = {}
        timeseries = metrics[TIMESERIES]
        metric_names = timeseries[TIMESERIES_METRIC_NAMES]

        if VALUE in metrics:
            metrics_to_publish[metric_names[TIMESERIES_METRIC_NAMES_VALUE]] = float(metrics[VALUE])

        if THRESHOLD in metrics:
            metrics_to_publish[metric_names[TIMESERIES_METRIC_NAMES_THRESHOLD]] = float(metrics[THRESHOLD])

        run_id = timeseries[TIMESERIES_RUN_ID]

        print(f"Publishing metric {metrics_to_publish} to run {run_id} at step {metric_step}.")
        self.runmetric_client.publish_metrics(
            run_id, metrics_to_publish, metric_step
        )
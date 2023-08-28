# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Builder class which creates a metrics object."""

from typing import List
from pyspark.sql import Row
import datetime
from shared_utilities.run_metrics_utils import publish_metrics


class RunMetricPublisher:
    """Builder class which creates a metrics object."""

    def __init__(self):
        """Construct a MetricOutputBuilder instance."""

    def publish_metrics(self, metrics: dict, metric_step: datetime):
        """Publish metrics to AML Run Metrics."""

        # check if run metrics is present at this level
        if("timeseries" in metrics):
            print("Publishing metrics.")
            self._publish_metric_to_run(metrics)

        # Iterate through nested groups and call publish metrics recursively
        if("groups" in metrics):
            for group in metrics["groups"]:
                self.publish_metrics(metrics[group])

    def _publish_metric_to_run(self, metrics: dict, metric_step: datetime):

        metrics_to_publish = {}
        timeseries = metrics["timeseries"]
        metrics_to_publish[timeseries["metricNames"]["value"]] = float(metrics["value"])

        if "threshold" in metrics:
            metrics_to_publish[timeseries["metricNames"]["threshold"]] = float(metrics["threshold"])

        run_id = timeseries["run_id"]

        publish_metrics(
            run_id, metrics_to_publish, metric_step
        )
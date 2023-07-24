# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Builder class which creates a metrics object."""

from typing import List
from pyspark.sql import Row


class MetricOutputBuilder:
    """Builder class which creates a metrics object."""

    def __init__(self, metrics: List[Row]):
        """Construct a MetricOutputBuilder instance."""
        metrics_dict = {}
        for metric in metrics:
            metric_name = metrics["metric_name"]
            node_id = metric["group"]

            if "Aggregate" in metric["group_pivot"]:
                # Add an aggregate metric.
                new_metric = {
                    node_id: {
                        "value": metric["metric_value"],
                        "threshold": metric["metric_threshold"],
                    }
                }
                if metric_name in metrics_dict:
                    metrics_dict[metric_name].update(new_metric)
                else:
                    metrics_dict[metric_name] = new_metric
            else:
                # Add a group-scoped metric.
                # TODO: Support nested groups.
                group_pivot = metric["group_pivot"]
                new_group_metric = {
                    metric["group_pivot"]: {
                        "value": metric["metric_value"],
                        "threshold": metric["metric_threshold"],
                    }
                }

                if metric_name in metrics_dict and node_id in metrics_dict[metric_name]:
                    metrics_dict[metric_name][node_id].update(new_group_metric)
                else:
                    metrics_dict[metric_name][node_id] = new_group_metric

        self.metric_dict = metrics_dict

    def get_metrics_object(self) -> dict:
        """Get the metrics object."""
        return self.metric_dict

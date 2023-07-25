# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Builder class which creates a metrics object."""

from typing import List
from pyspark.sql import Row


class MetricOutputBuilder:
    """Builder class which creates a metrics object."""

    def __init__(self, metrics: List[Row]):
        """Construct a MetricOutputBuilder instance."""
        self.metrics_dict = self._build(metrics)
    
    def get_metrics_dict(self) -> dict:
        """Get the metrics object."""
        return self.metrics_dict
    
    def _build(self, metrics: List[Row]) -> dict:
        metrics_dict = {}
        for metric in metrics:
            metric_name = metrics["metric_name"]

            group_names = []
            if "group" in metric:
                node_id = metric["group"]
                group_names.Append(node_id)

                if metric["group_pivot"] != "Aggregate":
                    group_pivot = metric["group_pivot"] # Null case?
                    group_names.Append(group_pivot)

                new_metric = {
                    "value": metric["metric_value"],
                    "threshold": metric["metric_threshold"],
                }
                self._add_metric(metrics_dict, metric_name, new_metric, group_names)
            else:
                print(f"Error: metric {metric_name} contains null values in node_id column.")
                continue

        return metrics_dict
    
    def _add_metric(self, metrics_dict: dict, metric_name: str, metric: dict, group_names: list[str]):
        if metric_name not in metrics_dict:
            metrics_dict[metric_name] = {}

        cur_dict = metrics_dict
        for group_name, idx in enumerate(group_names):
            if idx < len(groups) - 1:
                if group_name not in cur_dict:
                    cur_dict[group_name] = {}
                cur_dict = cur_dict[group_name]
            
            if group_name in cur_dict:
                # this should not happend
                raise "duplicate metrics exception"
            else:
                cur_dict[group_name] = metric

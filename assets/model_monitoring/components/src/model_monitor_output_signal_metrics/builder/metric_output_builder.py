# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Builder class which creates a metrics object."""

from typing import List
from pyspark.sql import Row

from shared_utilities.constants import (
    AGGREGATE,
    GROUP,
    GROUP_PIVOT,
    GROUPS,
    METRIC_NAME,
    METRIC_VALUE,
    THRESHOLD,
    THRESHOLD_VALUE,
    VALUE,
)


class MetricOutputBuilder:
    """Builder class which creates a metrics object."""

    def __init__(self, metrics: List[Row]):
        """Construct a MetricOutputBuilder instance."""
        self.metrics_dict = self._build(metrics)

    def get_metrics_dict(self) -> dict:
        """Get the metrics object."""
        return self.metrics_dict

    # Expected columns: group, group_pivot, metric_name, metric_value
    def _build(self, metrics: List[Row]) -> dict:
        metrics_dict = {}
        for metric in metrics:
            metric_dict = metric.asDict()
            metric_name = metric_dict[METRIC_NAME]

            group_names = []
            if GROUP in metric_dict:
                try:
                    node_id = metric_dict[GROUP]
                    group_names.append(node_id)

                    if metric_dict[GROUP_PIVOT].lower() != AGGREGATE:
                        group_pivot = metric_dict[GROUP_PIVOT]
                        group_names.append(group_pivot)

                    new_metric = {
                        VALUE: metric_dict[METRIC_VALUE],
                    }

                    if THRESHOLD_VALUE in metric_dict and metric_dict[THRESHOLD_VALUE] is not None:
                        new_metric[THRESHOLD] = metric_dict[THRESHOLD_VALUE]

                except Exception as e:
                    print(f"Error: Failed to get required column from metric '{metric_name}'"
                          + f" with exception: {str(e)}")
                    continue

                self._add_metric(metrics_dict, metric_name, new_metric, group_names)
            else:
                print(f"Error: The value of column '{GROUP}' in metric '{metric_name}' is null.")
                continue

        return metrics_dict

    def _add_metric(self, metrics_dict: dict, metric_name: str, metric: dict, group_names: List[str]):
        if metric_name not in metrics_dict:
            metrics_dict[metric_name] = {}

        cur_dict = metrics_dict[metric_name]
        for idx, group_name in enumerate(group_names):
            if idx < len(group_names) - 1:
                self._create_metric_groups_if_not_exist(cur_dict, group_name)
                cur_dict = cur_dict[GROUPS][group_name]
            else:
                if group_name in cur_dict:
                    raise Exception(f"Error: A duplicate metrics is found for metric {metric_name},"
                                    + f" group {group_names}")
                else:
                    self._create_metric_groups_if_not_exist(cur_dict, group_name)
                    cur_dict[GROUPS][group_name] = metric

    def _create_metric_groups_if_not_exist(self, cur_dict: dict, group_name: str):
        if GROUPS not in cur_dict:
            cur_dict[GROUPS] = {}
        if group_name not in cur_dict[GROUPS]:
            cur_dict[GROUPS][group_name] = {}

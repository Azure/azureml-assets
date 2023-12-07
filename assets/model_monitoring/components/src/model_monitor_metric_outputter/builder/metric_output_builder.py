# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Builder class which creates a metrics object."""

from typing import List
from pyspark.sql import Row

from runmetric_client import RunMetricClient
from shared_utilities.constants import (
    AGGREGATE,
    SIGNAL_METRICS_GROUP,
    SIGNAL_METRICS_GROUP_DIMENSION,
    GROUPS,
    SIGNAL_METRICS_METRIC_NAME,
    SIGNAL_METRICS_METRIC_VALUE,
    THRESHOLD,
    SIGNAL_METRICS_THRESHOLD_VALUE,
    VALUE,
    TIMESERIES,
    TIMESERIES_RUN_ID,
    TIMESERIES_METRIC_NAMES,
    TIMESERIES_METRIC_NAMES_VALUE,
    TIMESERIES_METRIC_NAMES_THRESHOLD,
)


class MetricOutputBuilder:
    """Builder class which creates a metrics object."""

    def __init__(self, runmetric_client: RunMetricClient, monitor_name: str, signal_name: str, metrics: List[Row]):
        """Construct a MetricOutputBuilder instance."""
        self.runmetric_client: RunMetricClient = runmetric_client
        self.metrics_dict = self._build(monitor_name, signal_name, metrics)

    def get_metrics_dict(self) -> dict:
        """Get the metrics object."""
        return self.metrics_dict

    # Expected columns: group, group_dimension, metric_name, metric_value
    def _build(self, monitor_name: str, signal_name: str, metrics: List[Row]) -> dict:
        metrics_dict = {}
        for metric in metrics:
            metric_dict = metric.asDict()

            try:
                metric_name = metric_dict[SIGNAL_METRICS_METRIC_NAME]

                group_names = []
                if SIGNAL_METRICS_GROUP in metric_dict and metric_dict[SIGNAL_METRICS_GROUP] is not None and metric_dict[SIGNAL_METRICS_GROUP] != "":  # noqa

                    root_group = metric_dict[SIGNAL_METRICS_GROUP]
                    group_names.append(root_group)

                    # If group dimension is not specified, default is Aggregate.
                    if (SIGNAL_METRICS_GROUP_DIMENSION in metric_dict
                            and metric_dict[SIGNAL_METRICS_GROUP_DIMENSION] is not None
                            and metric_dict[SIGNAL_METRICS_GROUP_DIMENSION] != ""
                            and metric_dict[SIGNAL_METRICS_GROUP_DIMENSION].lower() != AGGREGATE):
                        group_dimension = metric_dict[SIGNAL_METRICS_GROUP_DIMENSION]
                        group_names.append(group_dimension)

                new_metric = {
                    VALUE: metric_dict[SIGNAL_METRICS_METRIC_VALUE],
                }

                if (SIGNAL_METRICS_THRESHOLD_VALUE in metric_dict
                        and metric_dict[SIGNAL_METRICS_THRESHOLD_VALUE] is not None):
                    new_metric[THRESHOLD] = metric_dict[SIGNAL_METRICS_THRESHOLD_VALUE]

                self._add_metric(metrics_dict, monitor_name, signal_name, metric_name, new_metric, group_names)
            except Exception as e:
                print(f"Error: Failed to get required column from metric '{metric_name}'"
                      + f" with exception: {str(e)}")
                continue

        return metrics_dict

    def _add_metric(
            self,
            metrics_dict: dict,
            monitor_name: str,
            signal_name: str,
            metric_name: str,
            metric: dict,
            group_names: List[str]):
        if metric_name not in metrics_dict:
            metrics_dict[metric_name] = {}

        cur_dict = metrics_dict[metric_name]
        groups = group_names.copy()
        if len(group_names) > 0:
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
                        cur_dict[GROUPS][group_name][TIMESERIES] = self._create_timeseries(
                            monitor_name=monitor_name,
                            signal_name=signal_name,
                            metric_name=metric_name,
                            groups=groups)
        else:
            if VALUE in metric and metric[VALUE] is not None:
                cur_dict[VALUE] = metric[VALUE]
            if THRESHOLD in metric and metric[THRESHOLD] is not None:
                cur_dict[THRESHOLD] = metric[THRESHOLD]

            # Add run metrics
            cur_dict[TIMESERIES] = self._create_timeseries(
                monitor_name=monitor_name,
                signal_name=signal_name,
                metric_name=metric_name,
                groups=groups)

    def _create_timeseries(self, monitor_name: str, signal_name: str, metric_name: str, groups: List[str]):

        if groups is None:
            groups = []

        run_id = self.runmetric_client.get_or_create_run_id(
            monitor_name=monitor_name,
            signal_name=signal_name,
            metric_name=metric_name,
            groups=groups
        )

        print(f"Got run id {run_id} for metric {metric_name}, groups [{', '.join(groups)}],"
              + "signal {signal_name}, monitor {monitor_name}.")

        return {
            TIMESERIES_RUN_ID: run_id,
            TIMESERIES_METRIC_NAMES: {
                TIMESERIES_METRIC_NAMES_VALUE: "value",
                TIMESERIES_METRIC_NAMES_THRESHOLD: "threshold"
            }
        }

    def _create_metric_groups_if_not_exist(self, cur_dict: dict, group_name: str):
        if GROUPS not in cur_dict:
            cur_dict[GROUPS] = {}
        if group_name not in cur_dict[GROUPS]:
            cur_dict[GROUPS][group_name] = {}

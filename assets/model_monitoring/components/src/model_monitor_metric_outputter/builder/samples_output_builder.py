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
from shared_utilities.dict_utils import merge_dicts


class SamplesOutputBuilder:
    """Builder class which creates a samples object."""

    def __init__(self, runmetric_client: RunMetricClient, monitor_name: str, signal_name: str, samples_index: List[Row]):
        """Construct a SamplesOutputBuilder instance."""
        self.runmetric_client: RunMetricClient = runmetric_client

        self.samples_dict = self._build(samples_index)

    def get_samples_dict(self) -> dict:
        """Get the samples object."""
        return self.samples_dict

    # Expected columns: metric_name, group, group_dimension, samples_name, asset
    def _build(self, samples_index: List[Row]) -> dict:

        if not samples_index:
            return {}
        
        results = {}

        for sample in samples_index:
            sample_dict = sample.asDict()

            group_names = []

            group = self._get_group_from_dict(SIGNAL_METRICS_GROUP, sample_dict)
            if group:
                group_names.append(group)

            group_dimension = self._get_group_from_dict(SIGNAL_METRICS_GROUP_DIMENSION, sample_dict)            
            if group_dimension:
                group_names.append(group_dimension)
            
            results = merge_dicts(results, self._create_samples_entry(sample_dict, group_names))

        return results


    def _create_samples_entry(self, sample_row_dict : dict, group_names: List[str]):
            entry = {
                "samples": {
                    sample_row_dict["samples_name"]:
                    {
                        "uri": sample_row_dict["asset"]
                    }
                }
            }

            return {
                    sample_row_dict["metric_name"]: self._create_recursive_entry(group_names, entry)
            }


    def _create_recursive_entry(self, group_names: List[str], payload: dict):
        
        if len(group_names) > 0:
            return {
                "groups":
                {
                    group_names[0] : self._create_recursive_entry(group_names[1:], payload)
                }
            }

        return payload


    def _get_group_from_dict(self, group_key, dictionary: dict):
        if (group_key in dictionary
            and dictionary[group_key] is not None
            and dictionary[group_key] != ""):
            return dictionary[group_key]
        return None

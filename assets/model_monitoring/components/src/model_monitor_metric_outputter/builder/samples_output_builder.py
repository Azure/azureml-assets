# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Builder class which creates a samples object."""

from typing import List
from pyspark.sql import Row

from shared_utilities.constants import (
    ASSET_COLUMN,
    GROUPS,
    SAMPLES_COLUMN,
    SAMPLES_NAME_COLUMN,
    SAMPLES_URI,
    SIGNAL_METRICS_GROUP,
    SIGNAL_METRICS_GROUP_DIMENSION,
    SIGNAL_METRICS_METRIC_NAME,
)
from shared_utilities.dict_utils import merge_dicts


class SamplesOutputBuilder:
    """Builder class which creates a samples object."""

    def __init__(self, samples_index: List[Row]):
        """Construct a SamplesOutputBuilder instance."""
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

            group_dimension = self._get_group_from_dict(
                SIGNAL_METRICS_GROUP_DIMENSION, sample_dict
            )
            if group_dimension:
                group_names.append(group_dimension)

            results = merge_dicts(
                results, self._create_samples_entry(sample_dict, group_names)
            )

        return results

    def _create_samples_entry(self, sample_row_dict: dict, group_names: List[str]):
        entry = {
            SAMPLES_COLUMN: {
                sample_row_dict[SAMPLES_NAME_COLUMN]: {
                    SAMPLES_URI: sample_row_dict[ASSET_COLUMN]
                }
            }
        }

        return {
            sample_row_dict[SIGNAL_METRICS_METRIC_NAME]: self._create_recursive_entry(
                group_names, entry
            )
        }

    def _create_recursive_entry(self, group_names: List[str], payload: dict):

        if len(group_names) > 0:
            return {
                GROUPS: {
                    group_names[0]: self._create_recursive_entry(
                        group_names[1:], payload
                    )
                }
            }

        return payload

    def _get_group_from_dict(self, group_key, dictionary: dict):
        if (
            group_key in dictionary
            and dictionary[group_key] is not None
            and dictionary[group_key] != ""
        ):
            return dictionary[group_key]
        return None

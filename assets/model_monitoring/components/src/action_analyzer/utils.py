# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains utilities for action analyzer."""

from shared_utilities.io_utils import try_read_mltable_in_spark
from action_analyzer.constants import VIOLATED_METRICS_COLUMN

def get_unique_values_by_column(df, column):
    """Get the unique set for a given column."""
    unique_values = set()
    for data_row in df.collect():
        unique_values.add(data_row[column])
    return unique_values


def get_violated_metrics(violated_metrics_path):
    """Get violated metrics in a list."""
    violated_metrics_df = try_read_mltable_in_spark(
        violated_metrics_path, "violated_metrics"
    )
    return violated_metrics_df.select(VIOLATED_METRICS_COLUMN).rdd.flatMap(lambda x: x).collect()

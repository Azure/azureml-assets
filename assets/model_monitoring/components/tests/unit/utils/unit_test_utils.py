# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Unit test level helper utilities."""

from pyspark.sql import DataFrame


def assert_spark_dataframe_equal(actual_df: DataFrame, expected_df: DataFrame):
    """Assert two spark dataframes are equal."""
    assert actual_df.schema == expected_df.schema
    assert actual_df.count() == expected_df.count()
    actual_collected = actual_df.collect()
    expected_collected = expected_df.collect()
    print(f'Actual: {actual_collected}')
    print(f'Expected: {expected_collected}')
    assert actual_collected == expected_collected

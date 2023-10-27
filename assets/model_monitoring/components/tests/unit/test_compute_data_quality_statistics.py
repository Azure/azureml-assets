# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains unit tests for the Model Monitor Data Quality Compute Metric component."""

from pandas.testing import assert_frame_equal
from src.data_quality_statistics.compute_data_quality_statistics import (
    compute_data_quality_statistics,
    exclude_boolean_feature_from_df)
from tests.e2e.utils.io_utils import create_pyspark_dataframe
import pytest


df = [
        ("string1", 2, True, 4.67),
        ("string1", 3, False, 90.1),
        ("string1", 4, True, 2.8987),
        ("string1", 5, False, 3.454),
        ]
columns = ["feature_string", "feature_int", "feature_boolean", "feature_double"]
df = create_pyspark_dataframe(df, columns)

df_exclude_boolean = [
        ("string1", 2, 4.67),
        ("string1", 3, 90.1),
        ("string1", 4, 2.8987),
        ("string1", 5, 3.454),
        ]
columns = ["feature_string", "feature_int", "feature_double"]
df_exclude_boolean = create_pyspark_dataframe(df_exclude_boolean, columns)

data_stat_df = [
                ("feature_string",  None, None,   "StringType()",   "[string1]"),
                ("feature_int",     5.0,  2.0,    "LongType()",     None),
                ("feature_boolean", None, None,   "BooleanType()",  None),
                ("feature_double",  90.1, 2.8987, "DoubleType()",   None),
]
data_stat_colums = ["featureName", "max_value", "min_value", "dataType", "set"]

data_stats_table = create_pyspark_dataframe(data_stat_df, data_stat_colums)


@pytest.mark.unit
class TestModelMonitorDataQualityStatistic:
    """Test class for model monitor data quality statistics."""

    @pytest.mark.parametrize("df, data_stats_table",
                             [(df, data_stats_table)])
    def test_compute_data_quality_statistics(
            self,
            df,
            data_stats_table
    ):
        """Test compute data quality statistics with string, integer, boolean, double type."""
        actual_data_stats_table = compute_data_quality_statistics(df)
        assert_frame_equal(actual_data_stats_table.to_pandas(), data_stats_table.toPandas(), check_like=True)

    @pytest.mark.parametrize("df, df_exclude_boolean",
                             [(df, df_exclude_boolean)])
    def test_exclude_boolean_feature_from_df(
            self,
            df,
            df_exclude_boolean
    ):
        """Test exclude the boolean columns from dataframe."""
        actual_df_exclude_boolean = exclude_boolean_feature_from_df(df)
        assert_frame_equal(actual_df_exclude_boolean.toPandas(), df_exclude_boolean.toPandas())

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains unit tests for the Model Monitor Data Quality Compute Metric component."""

from pyspark.sql import functions as F
from pyspark.sql.types import (
    BinaryType,
    BooleanType,
    ByteType,
    CharType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    StructType,
    StructField,
    StringType,
    ShortType,
    TimestampType,
)
from pyspark_test import assert_pyspark_df_equal
from src.data_quality_statistics.compute_data_quality_statistics import (
    compute_data_quality_statistics,
    get_features_for_max_min_calculation)
from tests.e2e.utils.io_utils import create_pyspark_dataframe
import pytest


df = [
        ("string1", 2, True,  4.67, 4, "2023-10-01 00:00:01", 100, bytearray([10, 50]), 3.549999952316284,
         4, "char"),
        ("string1", 3, False, 90.1, 5, "2023-10-01 00:00:02", 200, bytearray([10, 50]), 3.55, 6, "char"),
        ("string1", 4, True,  2.8987, -1, "2023-10-01 00:00:03", 300, bytearray([10, 50]), 45.6, 7, "char"),
        ("string1", 5, False, 3.454, -2, "2023-10-01 00:00:04", 400, bytearray([10, 50]),  56.70000076293945,
         9, "char"),
        ]
schema = StructType([
    StructField("feature_string", StringType(), True),
    StructField("feature_int", IntegerType(), True),
    StructField("feature_boolean", BooleanType(), True),
    StructField("feature_double", DoubleType(), True),
    StructField("feature_byte", ByteType(), True),
    StructField("feature_timestamp", StringType(), True),
    StructField("feature_long", LongType(), True),
    StructField("feature_binary", BinaryType(), True),
    StructField("feature_float", FloatType(), True),
    StructField("feature_short", ShortType(), True),
    StructField("feature_char", StringType(), True),
    ]
)
df = create_pyspark_dataframe(df, schema)
df_with_timestamp = df.withColumn("feature_timestamp", df["feature_timestamp"].cast(TimestampType()))
df_with_timestamp = df_with_timestamp.withColumn("feature_char", df_with_timestamp["feature_char"].cast(CharType(30)))
df_with_timestamp = df_with_timestamp.withColumn("feature_date", F.current_date())

df_for_max_min_value = [
        (2, 4.67, 4, 100, 3.549999952316284),
        (3, 90.1, 5, 200, 3.55),
        (4, 2.8987, -1, 300, 45.6),
        (5, 3.454, -2, 400, 56.70000076293945),
        ]
schema = StructType([
    StructField("feature_int", IntegerType(), True),
    StructField("feature_double", DoubleType(), True),
    StructField("feature_byte", ByteType(), True),
    StructField("feature_long", LongType(), True),
    StructField("feature_float", FloatType(), True),
    ]
)
df_for_max_min_value = create_pyspark_dataframe(df_for_max_min_value, schema)

data_stat_df = [
                ("feature_string", None, None, "StringType()", "[string1]"),
                ("feature_int", 5.0,  2.0, "IntegerType()", None),
                ("feature_boolean", None, None, "BooleanType()", None),
                ("feature_double", 90.1, 2.8987, "DoubleType()", None),
                ("feature_byte", 5.0, -2.0, "ByteType()", None),
                ("feature_timestamp", None, None, "TimestampType()", None),
                ("feature_long", 400.0, 100.0, "LongType()", None),
                ("feature_binary", None, None, "BinaryType()", None),
                ("feature_float", 56.70000076293945, 3.549999952316284, "FloatType()", None),
                ("feature_short", None, None, "ShortType()", None),
                ("feature_char", None, None, "StringType()", "[char]"),
                ("feature_date", None, None, "DateType()", None),
]
data_stat_colums = ["featureName", "max_value", "min_value", "dataType", "set"]

data_stats_table = create_pyspark_dataframe(data_stat_df, data_stat_colums)


@pytest.mark.local
class TestModelMonitorDataQualityStatistic:
    """Test class for model monitor data quality statistics."""

    @pytest.mark.parametrize("df_with_timestamp, data_stats_table",
                             [(df_with_timestamp, data_stats_table)])
    def test_compute_data_quality_statistics(
            self,
            df_with_timestamp,
            data_stats_table
    ):
        """Test compute data quality statistics with string, integer, boolean, double type."""
        actual_data_stats_table = compute_data_quality_statistics(df_with_timestamp)
        assert data_stats_table.count() == actual_data_stats_table.to_spark().count()
        assert sorted(data_stats_table.collect()) == sorted(actual_data_stats_table.to_spark().collect())
        assert_pyspark_df_equal(data_stats_table, actual_data_stats_table.to_spark())

    @pytest.mark.parametrize("df_with_timestamp, df_for_max_min_value",
                             [(df_with_timestamp, df_for_max_min_value)])
    def test_get_features_for_max_min_calculation(
            self,
            df_with_timestamp,
            df_for_max_min_value
    ):
        """Test exclude the boolean columns from dataframe."""
        actual_df_for_max_min_value = get_features_for_max_min_calculation(df_with_timestamp)
        assert_pyspark_df_equal(df_for_max_min_value, actual_df_for_max_min_value)

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains unit tests for the Model Monitor Data Quality Compute Metric component."""

from pyspark.sql.functions import (
    current_timestamp,
    lit)
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType)
from pyspark_test import assert_pyspark_df_equal
from src.data_quality_compute_metrics.compute_data_quality_metrics import (
    compute_data_quality_metrics,
    compute_dtype_violation_count_modify_dataset,
    modify_dataType)
from tests.e2e.utils.io_utils import create_pyspark_dataframe
from tests.unit.test_compute_data_quality_statistics import df_with_timestamp, data_stats_table
import pytest


df_without_violation = [
        (True, 41.2, 101001, "2021-03-11", 2,
         345.66, 234, 45634, "string")
        ]
columns = ["feature_boolean", "feature_double", "feature_binary", "feature_date", "feature_byte",
           "feature_float", "feature_integer", "feature_long", "feature_string"]
df = create_pyspark_dataframe(df_without_violation, columns)
df_without_violation = df.withColumn("feature_timestamp", current_timestamp())

data_with_violation = [
        ("*", "*", "*", "*",
         "*", "*", "*", None),
        ("*", "*", "*", "*",
         "*", "*", "*", "*")
        ]
columns = ["feature_boolean", "feature_double",  "feature_date", "feature_short",
           "feature_float", "feature_integer", "feature_long", "feature_string"]
df_with_violation = create_pyspark_dataframe(data_with_violation, columns)

columns = ['featureName',  'dataType']
data_stats = [
            ('feature_timestamp', 'timestamp'),
            ('feature_boolean', 'boolean'),
            ('feature_double', 'double'),
            ('feature_binary', 'binary'),
            ('feature_date', 'date'),
            ('feature_decimal', 'decimal'),
            ('feature_float', 'float'),
            ('feature_integer', 'integer'),
            ('feature_long', 'long'),
            ('feature_string', 'string'),
        ]
data_stats_table_mod = create_pyspark_dataframe(data_stats, columns)


@pytest.mark.local
class TestModelMonitorDataQuality:
    """Test class for model monitor data quality component."""

    @pytest.mark.parametrize("df_without_datatype_violation, data_stats_table_mod",
                             [(df_without_violation, data_stats_table_mod)])
    def test_compute_dtype_violation_count_modify_dataset_without_violation(
            self,
            df_without_datatype_violation,
            data_stats_table_mod
    ):
        """Test datatype with no violation."""
        _, df_conversion_errors = compute_dtype_violation_count_modify_dataset(df_without_datatype_violation,
                                                                               data_stats_table_mod)
        for rows in df_conversion_errors.select("violationCount").collect():
            assert rows[0] == 0

    @pytest.mark.parametrize("df_with_datatype_violation, data_stats_table_mod",
                             [(df_with_violation, data_stats_table_mod)])
    def test_compute_dtype_violation_count_modify_dataset_with_violation(
            self,
            df_with_datatype_violation,
            data_stats_table_mod
    ):
        """Test datatype while there are violation."""
        _, df_conversion_errors = compute_dtype_violation_count_modify_dataset(df_with_datatype_violation,
                                                                               data_stats_table_mod)
        for rows in df_conversion_errors.select("violationCount").collect():
            assert rows[0] >= 1

    def test_compute_dtype_violation_count_modify_dataset_with_unsupported_type(self):
        """Test datatype while the datatype is unsupported."""
        data = [(1, 2), (2, 3)]
        columns = ["feature_unsupported_1", "feature_unsupported_2"]
        df = create_pyspark_dataframe(data, columns)
        data_stats_tabel_mod_with_unsupported_type = [("feature_unsupported_1", "UnsupportedType()"),
                                                      ("feature_unsupported_2", "UnsupportedType()")]
        columns = ['featureName',  'dataType']
        data_stats_table_mod = create_pyspark_dataframe(data_stats_tabel_mod_with_unsupported_type, columns)
        _, df_conversion_errors = compute_dtype_violation_count_modify_dataset(df, data_stats_table_mod)
        for rows in df_conversion_errors.select("violationCount").collect():
            assert rows[0] == 0

    def test_modify_type(self):
        """Test modify datatype from Datatype() to datatype."""
        data = [("BinaryType()", "featureName1"),  ("TimestampType()", "featureName1"),
                ("BooleanType()", "featureName1"), ("DoubleType()", "featureName1"),
                ("StringType()", "featureName1"),  ("DateType()", "featureName1"),
                ("LongType()", "featureName1"), ("ByteType()", "featureName1"),
                ("IntegerType()", "featureName1")]
        columns = ["dataType", "featureName"]
        expected = ["binary",      "timestamp", "boolean",
                    "double",      "string",    "date",
                    "long",        "byte",      "integer"]
        df = create_pyspark_dataframe(data, columns)
        df_mod = modify_dataType(df)
        dataType_array = [str(row.dataType) for row in df_mod.collect()]
        assert dataType_array == expected

    def test_compute_data_quality_metrics(self):
        """Test compute data quality metrics."""
        expected_metrics_data = [
                ("feature_short",     "DataTypeErrorRate",  "Numerical",     0,   0.0),
                ("feature_timestamp", "NullValueRate",      "Categorical",   0,   0.0),
                ("feature_binary",    "NullValueRate",      "Categorical",   0,   0.0),
                ("feature_char",      "NullValueRate",      "Categorical",   0,   0.0),
                ("feature_boolean",   "DataTypeErrorRate",  "Categorical",   0,   0.0),
                ("feature_double",    "NullValueRate",      "Numerical",     0,   0.0),
                ("feature_float",     "NullValueRate",      "Numerical",     0,   0.0),
                ("feature_boolean",   "NullValueRate",      "Categorical",   0,   0.0),
                ("feature_binary",    "DataTypeErrorRate",  "Categorical",   0,   0.0),
                ("feature_double",    "DataTypeErrorRate",  "Numerical",     0,   0.0),
                ("feature_timestamp", "DataTypeErrorRate",  "Categorical",   0,   0.0),
                ("feature_string",    "DataTypeErrorRate",  "Categorical",   0,   0.0),
                ("feature_string",    "NullValueRate",      "Categorical",   0,   0.0),
                ("feature_int",       "DataTypeErrorRate",  "Numerical",     0,   0.0),
                ("feature_char",      "DataTypeErrorRate",  "Categorical",   0,   0.0),
                ("feature_float",     "DataTypeErrorRate",  "Numerical",     0,   0.0),
                ("feature_short",     "NullValueRate",      "Numerical",     0,   0.0),
                ("feature_date",      "NullValueRate",      "Categorical",   0,   0.0),
                ("feature_long",      "OutOfBoundsRate",    "Numerical",     0,   0.0),
                ("feature_date",      "DataTypeErrorRate",  "Categorical",   0,   0.0),
                ("feature_byte",      "OutOfBoundsRate",    "Categorical",   0,   0.0),
                ("feature_byte",      "NullValueRate",      "Categorical",   0,   0.0),
                ("feature_byte",      "DataTypeErrorRate",  "Categorical",   0,   0.0),
                ("feature_char",      "OutOfBoundsRate",    "Categorical",   0,   0.0),
                ("feature_long",      "NullValueRate",      "Numerical",     0,   0.0),
                ("feature_float",     "OutOfBoundsRate",    "Numerical",     0,   0.0),
                ("feature_long",      "DataTypeErrorRate",  "Numerical",     0,   0.0),
                ("feature_string",    "OutOfBoundsRate",    "Categorical",   0,   0.0),
                ("feature_int",       "OutOfBoundsRate",    "Numerical",     0,   0.0),
                ("feature_double",    "OutOfBoundsRate",    "Numerical",     0,   0.0),
                ("feature_int",       "NullValueRate",      "Numerical",     0,   0.0),
        ]
        expected_metrics_data_columns = ["featureName", "metricName", "dataType", "violationCount", "metricValue"]
        expected_metrics_df = create_pyspark_dataframe(expected_metrics_data, expected_metrics_data_columns)
        row = create_pyspark_dataframe(
            [(0, "RowCount", 4.0)],
            StructType(
                [
                    StructField("violationCount", IntegerType(), True),
                    StructField("metricName", StringType(), True),
                    StructField("metricValue", DoubleType(), True)
                ])
            ).withColumn("featureName", lit("")).withColumn("dataType", lit(""))

        expected_metrics_df = expected_metrics_df.unionByName(row)
        metrics_df = compute_data_quality_metrics(df_with_timestamp, data_stats_table)
        assert expected_metrics_df.count() == metrics_df.count()
        assert sorted(expected_metrics_df.collect()) == sorted(metrics_df.collect())
        assert_pyspark_df_equal(expected_metrics_df, metrics_df, check_columns_in_order=False)

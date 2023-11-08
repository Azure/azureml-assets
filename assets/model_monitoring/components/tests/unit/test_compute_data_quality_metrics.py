# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains unit tests for the Model Monitor Data Quality Compute Metric component."""

import pytest
from pyspark.sql.functions import current_timestamp
from src.data_quality_compute_metrics.compute_data_quality_metrics import (
    compute_dtype_violation_count_modify_dataset,
    modify_dataType)
from tests.e2e.utils.io_utils import create_pyspark_dataframe


df_without_violation = [
        (True, 41.2, 101001, "2021-03-11", 2.345,
         345.66, 234, 45634, 3877, "string",
         "a", "a")
        ]
columns = ["feature_boolean", "feature_double", "feature_binary", "feature_date", "feature_decimal",
           "feature_float", "feature_integer", "feature_long", "feature_short", "feature_string",
           "feature_char", "feature_varchar"]
df = create_pyspark_dataframe(df_without_violation, columns)
df_without_violation = df.withColumn("feature_timestamp", current_timestamp())

data_with_violation = [
        ("*", "*", "*", "*",
         "*", "*", "*", "*", None,
         None, None),
        ("*", "*", "*", "*",
         "*", "*", "*", "*", 1,
         1, 1)
        ]
columns = ["feature_boolean", "feature_double",  "feature_date", "feature_decimal",
           "feature_float", "feature_integer", "feature_long", "feature_short", "feature_string",
           "feature_char", "feature_varchar"]
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
            ('feature_short', 'short'),
            ('feature_string', 'string'),
            ('feature_char', 'char(30)'),
            ('feature_varchar', 'varchar(30)')
        ]
data_stats_table_mod = create_pyspark_dataframe(data_stats, columns)


@pytest.mark.unit
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

    @pytest.mark.local
    def test_modify_type(self):
        """Test modify datatype from Datatype() to datatype."""
        data = [("BinaryType()", "featureName1"),  ("TimestampType()", "featureName1"),
                ("BooleanType()", "featureName1"), ("DoubleType()", "featureName1"),
                ("StringType()", "featureName1"),  ("DateType()", "featureName1"),
                ("LongType()", "featureName1"),    ("ShortType()", "featureName1"),
                ("CharType()", "featureName1"),    ("VarcharType()", "featureName1"),
                ("ByteType()", "featureName1"),    ("IntegerType()", "featureName1")]

        columns = ["dataType", "featureName"]
        expected = ["binary",      "timestamp", "boolean",
                    "double",      "string",    "date",
                    "long",        "short",     "char(30)",
                    "varchar(30)", "byte",      "integer"]
        df = create_pyspark_dataframe(data, columns)
        df_mod = modify_dataType(df)
        dataType_array = [str(row.dataType) for row in df_mod.collect()]
        assert dataType_array == expected

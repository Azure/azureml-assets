# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains unit tests for the Model Monitor Data Quality Compute Metric component."""

from pyspark.sql.functions import (
    current_timestamp,
    lit)
from pyspark.sql.types import (
    BooleanType,
    ByteType,
    DoubleType,
    IntegerType,
    FloatType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType)
from src.data_quality_compute_metrics.compute_data_quality_metrics import (
    compute_data_quality_metrics,
    compute_dtype_violation_count_modify_dataset,
    get_null_count,
    modify_dataType,
    compute_set_violation,
    impute_numericals_with_median,
    impute_categorical_with_mode,
    modify_categorical_columns)
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

    def test_modify_type(self):
        """Test modify datatype from Datatype() to datatype."""
        data = [("BinaryType()", "featureName1"),  ("TimestampType()", "featureName1"),
                ("BooleanType()", "featureName1"), ("DoubleType()", "featureName1"),
                ("StringType()", "featureName1"),  ("DateType()", "featureName1"),
                ("LongType()", "featureName1"), ("ByteType()", "featureName1"),
                ("IntegerType()", "featureName1"), ("ShortType()", "featureName1")]
        columns = ["dataType", "featureName"]
        expected = ["binary",      "timestamp", "boolean",
                    "double",      "string",    "date",
                    "bigint",      "tinyint",   "int",
                    "smallint"]
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
                ("feature_long",      "OutOfBoundsRate",    "Numerical",     0,   0.0),
                ("feature_byte",      "OutOfBoundsRate",    "Numerical",     0,   0.0),
                ("feature_byte",      "NullValueRate",      "Numerical",     0,   0.0),
                ("feature_byte",      "DataTypeErrorRate",  "Numerical",     0,   0.0),
                ("feature_char",      "OutOfBoundsRate",    "Categorical",   0,   0.0),
                ("feature_long",      "NullValueRate",      "Numerical",     0,   0.0),
                ("feature_float",     "OutOfBoundsRate",    "Numerical",     0,   0.0),
                ("feature_long",      "DataTypeErrorRate",  "Numerical",     0,   0.0),
                ("feature_string",    "OutOfBoundsRate",    "Categorical",   0,   0.0),
                ("feature_int",       "OutOfBoundsRate",    "Numerical",     0,   0.0),
                ("feature_double",    "OutOfBoundsRate",    "Numerical",     0,   0.0),
                ("feature_int",       "NullValueRate",      "Numerical",     0,   0.0),
                ("feature_short",     "OutOfBoundsRate",    "Numerical",     0,   0.0),
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
        metrics_df = compute_data_quality_metrics(df_with_timestamp, data_stats_table, None, None)
        assert expected_metrics_df.count() == metrics_df.count()
        assert sorted(expected_metrics_df.collect()) == sorted(metrics_df.collect())

    def test_get_null_count(self):
        """Test null count metrics."""
        df_for_null_value = [
                (2, 4.67, 4, 100, 3.549999952316284, "string"),
                (3, 90.1, 5, 200, 3.55, "string"),
                (4, 2.8987, -1, 300, 45.6, "string"),
                (5, 3.454, -2, 400, 56.70000076293945, "string"),
                (None, None, None, None, None, None)
        ]
        schema = StructType([
            StructField("feature_int", IntegerType(), True),
            StructField("feature_double", DoubleType(), True),
            StructField("feature_byte", ByteType(), True),
            StructField("feature_long", LongType(), True),
            StructField("feature_float", FloatType(), True),
            StructField("feature_string", StringType(), True),
        ])
        df_for_max_min_value = create_pyspark_dataframe(df_for_null_value, schema)
        expected_max_and_min_value_data = [("feature_int", 1, "NullValue"),
                                           ("feature_double", 1, "NullValue"),
                                           ("feature_byte", 1, "NullValue"),
                                           ("feature_long", 1, "NullValue"),
                                           ("feature_float", 1, "NullValue"),
                                           ("feature_string", 1, "NullValue")]
        data_schema = StructType(
            [
                StructField("featureName", StringType(), True),
                StructField("violationCount", IntegerType(), True),
                StructField("metricName", StringType(), True),
            ]
        )
        expected_metrics_df = create_pyspark_dataframe(expected_max_and_min_value_data, data_schema)
        metrics_df = get_null_count(df_for_max_min_value)
        assert expected_metrics_df.count() == metrics_df.count()
        assert sorted(expected_metrics_df.collect()) == sorted(metrics_df.collect())

    def test_impute_missing_values(self):
        """Test impute_numericals_with_median and impute_categorical_with_mode."""
        df_with_missing_value = [
            (40.1,   "s1",   1),
            (40.2,   "s1",   2),
            (None,   "s2",   3),
            (None,   "s1",   None),
            (40.3,   None,   None)
            ]
        df_with_inputed_value = [
            (40.1,   "s1",   1),
            (40.2,   "s1",   2),
            (40.2,   "s2",   3),
            (40.2,   "s1",   2),
            (40.3,   "s1",   2)
            ]
        schema = StructType([
            StructField("feature_double", DoubleType(), True),
            StructField("feature_string", StringType(), True),
            StructField("feature_integer", IntegerType(), True),
        ])

        df_with_missing_value = create_pyspark_dataframe(df_with_missing_value, schema)
        df_with_inputed_value = create_pyspark_dataframe(df_with_inputed_value, schema)

        numerical_columns = ["feature_double", "feature_integer"]
        categorical_columns = ["feature_string"]
        df_inputed_numerical = impute_numericals_with_median(df_with_missing_value, numerical_columns)
        df_inputed_categorical = impute_categorical_with_mode(df_inputed_numerical, categorical_columns)
        assert sorted(df_with_inputed_value.collect()) == sorted(df_inputed_categorical.collect())

    def test_compute_set_violation(self):
        """Test compute_set_violation."""
        df = [
                ("string1", "char", 2, True,  4.67, "2023-10-01 00:00:01"),
                ("string2", "char", 3, False, 90.1, "2023-10-01 00:00:02"),
                ("string3", "char", 4, True,  2.8987, "2023-10-01 00:00:03"),
                ("string4", "char", 5, False, 3.454, "2023-10-01 00:00:04")
        ]
        schema = StructType(
            [
                StructField("feature_string", StringType(), True),
                StructField("feature_char", StringType(), True),
                StructField("feature_int", IntegerType(), True),
                StructField("feature_boolean", BooleanType(), True),
                StructField("feature_double", DoubleType(), True),
                StructField("feature_timestamp", StringType(), True),
            ]
        )
        df = create_pyspark_dataframe(df, schema)
        df_input = df.withColumn("feature_timestamp", df["feature_timestamp"].cast(TimestampType()))

        expected_set_violation_value = [("feature_string", 1, "setValueOutOfRange"),
                                        ("feature_char", 0, "setValueOutOfRange")]
        data_schema = StructType(
            [
                StructField("featureName", StringType(), True),
                StructField("violationCount", IntegerType(), True),
                StructField("metricName", StringType(), True),
            ]
        )
        expected_set_violation_table = create_pyspark_dataframe(expected_set_violation_value, data_schema)

        set_violation_table = compute_set_violation(df_input, data_stats_table, ["feature_string", "feature_char"])
        assert sorted(expected_set_violation_table.collect()) == sorted(set_violation_table.collect())

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""This file contains unit tests for the df utilities."""

from pyspark.sql.types import (
    DoubleType,
    FloatType,
    StructField,
    StructType)
from pyspark.sql import SparkSession
from src.shared_utilities.df_utils import (
    get_common_columns,
    try_get_common_columns_with_error,
    try_get_common_columns,
    get_feature_type_override_map,
    is_numerical,
    is_categorical,
    get_numerical_cols_with_df_with_override,
    get_categorical_cols_with_df_with_override,
    get_numerical_and_categorical_cols,
    modify_categorical_columns
)
from src.shared_utilities.momo_exceptions import InvalidInputError
from tests.e2e.utils.io_utils import create_pyspark_dataframe
from tests.unit.test_compute_data_quality_statistics import df_with_timestamp
import pandas as pd
import pytest
import datetime


@pytest.mark.unit
class TestDFUtils:
    """Test class for df utilities."""

    def test_get_common_columns(self):
        """Test get common columns."""
        # Test with two empty dataframes
        spark = SparkSession.builder.appName("test").getOrCreate()
        emp_RDD = spark.sparkContext.emptyRDD()
        # Create empty schema
        columns = StructType([])

        # Create an empty RDD with empty schema
        baseline_df = create_pyspark_dataframe(emp_RDD, columns)
        production_df = create_pyspark_dataframe(emp_RDD, columns)
        assert get_common_columns(baseline_df, production_df) == {}

        # Test with two dataframes that have common columns but datatype is not in the same type
        float_data = [(3.55,), (6.88,), (7.99,)]
        schema = StructType([
            StructField("target", FloatType(), True)])
        baseline_df = create_pyspark_dataframe(float_data, schema)
        double_data = [(3.55,), (6.88,), (7.99,)]
        schema = StructType([
            StructField("target", DoubleType(), True)])
        production_df = create_pyspark_dataframe(double_data, schema)
        assert get_common_columns(baseline_df, production_df) == {'target': 'double'}

        # Test with two dataframes that have no common columns
        baseline_df = create_pyspark_dataframe([(1, "a"), (2, "b")],
                                               ["id", "name"])
        production_df = create_pyspark_dataframe([(3, "c"), (4, "d")],
                                                 ["age", "gender"])
        assert get_common_columns(baseline_df, production_df) == {}

        # Test with two dataframes that have one common column
        baseline_df = create_pyspark_dataframe([(1, "a"), (2, "b")],
                                               ["id", "name"])
        production_df = create_pyspark_dataframe([(1, "c"), (2, "d")],
                                                 ["id", "age"])
        assert get_common_columns(baseline_df, production_df) == {"id": "bigint"}

        # Test with two dataframes that have multiple common columns
        baseline_df = create_pyspark_dataframe([(1, "a", 10), (2, "b", 20)],
                                               ["id", "name", "age"])
        production_df = create_pyspark_dataframe([(1, "c", 30), (2, "d", 40)],
                                                 ["id", "name", "age"])
        assert get_common_columns(baseline_df, production_df) == {"id": "bigint", "name": "string", "age": "bigint"}

        # Test with two dataframes that have different Types in common columns
        baseline_df = create_pyspark_dataframe([(1.0, "a", 10), (2.0, "b", 20)],
                                               ["id", "name", "age"])
        production_df = create_pyspark_dataframe([(1, "c", 30), (2, "d", 40)],
                                                 ["id", "name", "age"])
        assert get_common_columns(baseline_df, production_df) == {"name": "string", "age": "bigint"}

    def test_get_feature_type_override_map(self):
        """Test get feature type override map."""
        # Test with no type override features
        override_numerical_features = None
        override_categorical_features = None
        assert get_feature_type_override_map(override_numerical_features, override_categorical_features) == {}

        # Test with only numerical override features
        override_numerical_features = "UserId,TransactionId"
        override_categorical_features = None
        assert get_feature_type_override_map(override_numerical_features,
                                             override_categorical_features) == {"UserId": "numerical",
                                                                                "TransactionId": "numerical"}

        # Test with only categorical override features
        override_numerical_features = None
        override_categorical_features = "Email"
        assert get_feature_type_override_map(override_numerical_features,
                                             override_categorical_features) == {"Email": "categorical"}

        # Test with both numerical and categorical override features
        override_numerical_features = "Age,Id"
        override_categorical_features = "Name,Gender"
        assert get_feature_type_override_map(override_numerical_features,
                                             override_categorical_features) == {"Age": "numerical",
                                                                                "Id": "numerical",
                                                                                "Name": "categorical",
                                                                                "Gender": "categorical"}

    def test_is_numerical(self):
        """Test is numerical."""
        spark = self.init_spark()
        column_dtype_map = {
                            'col1': 'int',
                            'col2': 'float',
                            'col3': 'double',
                            'col4': 'decimal',
                            'col5': 'string',
                            'col6': 'boolean',
                            'col7': 'timestamp',
                            'col8': 'date',
                            'col9': 'unknown',
                            'col10': 'fake'
                            }
        baseline_df = pd.DataFrame({'col1': [1, 2, 2, 4, 4]})
        baseline_df = spark.createDataFrame(baseline_df)

        # Test feature type override take priority
        feature_type_override_map = {'col2': 'categorical'}
        assert is_numerical('col2', column_dtype_map, feature_type_override_map, baseline_df) is False
        feature_type_override_map = {'col2': 'numerical'}
        assert is_numerical('col2', column_dtype_map, feature_type_override_map, baseline_df) is True

        # Test dtype_map take priority
        assert is_numerical('col2', column_dtype_map, {}, baseline_df) is True
        assert is_numerical('col3', column_dtype_map, {}, baseline_df) is True
        assert is_numerical('col4', column_dtype_map, {}, baseline_df) is True
        assert is_numerical('col5', column_dtype_map, {}, baseline_df) is False
        assert is_numerical('col6', column_dtype_map, {}, baseline_df) is False
        assert is_numerical('col7', column_dtype_map, {}, baseline_df) is False
        assert is_numerical('col8', column_dtype_map, {}, baseline_df) is False

        # Test with integer column with high distinct ratio
        baseline_df = pd.DataFrame({'col1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        baseline_df = spark.createDataFrame(baseline_df)
        assert is_numerical('col1', column_dtype_map, {}, baseline_df) is True

        # Test with integer column with low distinct value ratio
        baseline_df = pd.DataFrame({'col1': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                             1, 1, 1, 1, 1, None, 1, 1, 1, 1, 1, 1,
                                             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]})
        baseline_df = spark.createDataFrame(baseline_df)
        assert is_numerical('col1', column_dtype_map, {}, baseline_df) is False

        # Test with unknown data type
        assert is_numerical('col9', column_dtype_map, {}, baseline_df) is None
        assert is_numerical('col10', column_dtype_map, {}, baseline_df) is None

    def test_is_categorical(self):
        """Test is categorical."""
        spark = self.init_spark()
        column_dtype_map = {
                            'col1': 'int',
                            'col2': 'float',
                            'col3': 'double',
                            'col4': 'decimal',
                            'col5': 'string',
                            'col6': 'boolean',
                            'col7': 'timestamp',
                            'col8': 'date',
                            'col9': 'unknown',
                            'col10': 'fake'
                            }
        baseline_df = pd.DataFrame({'col1': [1, 2, 2, 4, 4]})
        baseline_df = spark.createDataFrame(baseline_df)

        # Test feature type override take priority
        feature_type_override_map = {'col2': 'categorical'}
        assert is_categorical('col2', column_dtype_map, feature_type_override_map, baseline_df) is True
        feature_type_override_map = {'col2': 'numerical'}
        assert is_categorical('col2', column_dtype_map, feature_type_override_map, baseline_df) is False

        # Test dtype_map take priority
        assert is_categorical('col2', column_dtype_map, {}, baseline_df) is False
        assert is_categorical('col3', column_dtype_map, {}, baseline_df) is False
        assert is_categorical('col4', column_dtype_map, {}, baseline_df) is False
        assert is_categorical('col5', column_dtype_map, {}, baseline_df) is True
        assert is_categorical('col6', column_dtype_map, {}, baseline_df) is True
        assert is_categorical('col7', column_dtype_map, {}, baseline_df) is True
        assert is_categorical('col8', column_dtype_map, {}, baseline_df) is True

        # Test with integer column with high distinct ratio
        baseline_df = pd.DataFrame({'col1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        baseline_df = spark.createDataFrame(baseline_df)
        assert is_categorical('col1', column_dtype_map, {}, baseline_df) is False

        # Test with integer column with low distinct value ratio
        baseline_df = pd.DataFrame({'col1': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                             1, 1, 1, 1, 1, None, 1, 1, 1, 1, 1, 1,
                                             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]})
        baseline_df = spark.createDataFrame(baseline_df)
        assert is_categorical('col1', column_dtype_map, {}, baseline_df) is True

        # Test with unknown data type
        assert is_categorical('col9', column_dtype_map, {}, baseline_df) is None
        assert is_categorical('col10', column_dtype_map, {}, baseline_df) is None

    def test_get_numerical_cols_with_df_with_override(self):
        """Test get numerical columns with data type override."""
        spark = self.init_spark()
        baseline_df = pd.DataFrame({
                            'col1': [1, 2, 2, 4, 4],
                            'col2': [1.1, 2.2, 3.3, 4.4, 5.5],
                            'col3': [1.11, 2.22, 3.33, 4.44, 5.55],
                            'col4': [1.111, 2.222, 3.333, 4.444, 5.555],
                            'col5': ['1', '2', '3', '4', '5'],
                            'col6': [True, True, False, False, True],
                            'col7': [bytes(55), bytes(33), bytes(22), bytes(55), bytes(33)],
                            'col8': [pd.Timestamp('2018-04-24 01:00:00'),
                                     pd.Timestamp('2019-04-24 02:00:00'),
                                     pd.Timestamp('2020-04-24 03:00:00'),
                                     pd.Timestamp('2021-04-24 04:00:00'),
                                     pd.Timestamp('2022-04-24 05:00:00')],
                            'col9': [datetime.date(2020, 5, 17),
                                     datetime.date(2021, 5, 17),
                                     datetime.date(2022, 5, 17),
                                     datetime.date(2023, 5, 17),
                                     datetime.date(2024, 5, 17)],
                            'col10': ['unknown', 'unknown', 'unknown', 'unknown', 'unknown']})
        baseline_df = spark.createDataFrame(baseline_df)

        # Test without dtype map without feature type override
        assert get_numerical_cols_with_df_with_override(baseline_df, None, None) == ['col1', 'col2', 'col3', 'col4']

        # Test with dtype map without feature type override
        column_dtype_map = {
                    'col1': 'int',
                    'col2': 'float',
                    'col3': 'double',
                    'col4': 'decimal',
                    'col5': 'decimal',
                    'col6': 'boolean',
                    'col7': 'binary',
                    'col8': 'timestamp',
                    'col9': 'date',
                    'col10': 'unknown'
                    }
        assert get_numerical_cols_with_df_with_override(baseline_df,
                                                        None,
                                                        None,
                                                        column_dtype_map) == ['col1', 'col2', 'col3',
                                                                              'col4', 'col5']

        # Test with feature type override
        numerical_features = 'col5,col6,col7,col8,col9'
        assert get_numerical_cols_with_df_with_override(baseline_df,
                                                        override_numerical_features=numerical_features,
                                                        override_categorical_features=None,
                                                        column_dtype_map=column_dtype_map) == ['col1', 'col2', 'col3',
                                                                                               'col4', 'col5', 'col6',
                                                                                               'col7', 'col8', 'col9']

    def test_get_categorical_cols_with_df_with_override(self):
        """Test get categorical columns with data type override."""
        spark = self.init_spark()
        baseline_df = pd.DataFrame({
                            'col1': [1, 2, 2, 4, 4],
                            'col2': [1.1, 2.2, 3.3, 4.4, 5.5],
                            'col3': [1.11, 2.22, 3.33, 4.44, 5.55],
                            'col4': [1.111, 2.222, 3.333, 4.444, 5.555],
                            'col5': ['1', '2', '3', '4', '5'],
                            'col6': [True, True, False, False, True],
                            'col7': [bytes(55), bytes(33), bytes(22), bytes(55), bytes(33)],
                            'col8': [pd.Timestamp('2018-04-24 01:00:00'),
                                     pd.Timestamp('2019-04-24 02:00:00'),
                                     pd.Timestamp('2020-04-24 03:00:00'),
                                     pd.Timestamp('2021-04-24 04:00:00'),
                                     pd.Timestamp('2022-04-24 05:00:00')],
                            'col9': [datetime.date(2020, 5, 17),
                                     datetime.date(2021, 5, 17),
                                     datetime.date(2022, 5, 17),
                                     datetime.date(2023, 5, 17),
                                     datetime.date(2024, 5, 17)],
                            'col10': ['unknown', 'unknown', 'unknown', 'unknown', 'unknown']})
        baseline_df = spark.createDataFrame(baseline_df)

        # Test without dtype map without feature type override
        assert get_categorical_cols_with_df_with_override(baseline_df, None, None) == ['col5', 'col6', 'col7',
                                                                                       'col8', 'col9', 'col10']

        # Test with dtype map without feature type override
        column_dtype_map = {
                    'col1': 'int',
                    'col2': 'float',
                    'col3': 'double',
                    'col4': 'decimal',
                    'col5': 'decimal',
                    'col6': 'boolean',
                    'col7': 'binary',
                    'col8': 'timestamp',
                    'col9': 'date',
                    'col10': 'unknown'
                    }
        assert get_categorical_cols_with_df_with_override(baseline_df,
                                                          None,
                                                          None,
                                                          column_dtype_map) == ['col6', 'col7',
                                                                                'col8', 'col9']

        # Test with feature type override
        categorical_features = 'col1,col2,col3,col4,col5'
        assert get_categorical_cols_with_df_with_override(baseline_df,
                                                          override_numerical_features=None,
                                                          override_categorical_features=categorical_features,
                                                          column_dtype_map=column_dtype_map) == ['col1', 'col2',
                                                                                                 'col3', 'col4',
                                                                                                 'col5', 'col6',
                                                                                                 'col7', 'col8',
                                                                                                 'col9']

    def test_get_numerical_and_categorical_cols(self):
        """Test get numerical and categorical columns with data type override."""
        spark = self.init_spark()
        baseline_df = pd.DataFrame({
                            'col1': [1, 2, 2, 4, 4],
                            'col2': [1.1, 2.2, 3.3, 4.4, 5.5],
                            'col3': [1.11, 2.22, 3.33, 4.44, 5.55],
                            'col4': [1.111, 2.222, 3.333, 4.444, 5.555],
                            'col5': ['1', '2', '3', '4', '5'],
                            'col6': [True, True, False, False, True],
                            'col7': [bytes(55), bytes(33), bytes(22), bytes(55), bytes(33)],
                            'col8': [pd.Timestamp('2018-04-24 01:00:00'),
                                     pd.Timestamp('2019-04-24 02:00:00'),
                                     pd.Timestamp('2020-04-24 03:00:00'),
                                     pd.Timestamp('2021-04-24 04:00:00'),
                                     pd.Timestamp('2022-04-24 05:00:00')],
                            'col9': [datetime.date(2020, 5, 17),
                                     datetime.date(2021, 5, 17),
                                     datetime.date(2022, 5, 17),
                                     datetime.date(2023, 5, 17),
                                     datetime.date(2024, 5, 17)]})
        baseline_df = spark.createDataFrame(baseline_df)
        numerical_columns, categorical_columns = get_numerical_and_categorical_cols(baseline_df, None, None)
        assert numerical_columns == ['col1', 'col2', 'col3', 'col4']
        assert categorical_columns == ['col5', 'col6', 'col7', 'col8', 'col9']

    def test_modify_categorical_columns(self):
        """Test modify_categorical_columns."""
        categorical_columns = ["feature_boolean", "feature_binary", "feature_timestamp",
                               "feature_string", "feature_int"]
        expected_categorical_columns = ["feature_string"]

        modified_categorical_columns = modify_categorical_columns(df_with_timestamp, categorical_columns)
        assert expected_categorical_columns == modified_categorical_columns

    def init_spark(self):
        """Get or create spark session."""
        spark = SparkSession.builder.appName("test").getOrCreate()
        return spark

    def test_try_get_common_columns_error(self):
        """Test scenarios for common_columns with error."""
        # Test with two empty dataframes
        spark = SparkSession.builder.appName("test").getOrCreate()
        emp_RDD = spark.sparkContext.emptyRDD()
        # Create empty schema
        columns = StructType([])

        # Create an empty RDD with empty schema
        baseline_df = create_pyspark_dataframe(emp_RDD, columns)
        production_df = create_pyspark_dataframe(emp_RDD, columns)
        with pytest.raises(Exception) as ex:
            try_get_common_columns_with_error(baseline_df, production_df)
        assert "Found no common columns between input datasets." in str(ex.value)

        # Test with two dataframes that have common columns but datatype is not in the same type
        float_data = [(3.55,), (6.88,), (7.99,)]
        schema = StructType([
            StructField("target", FloatType(), True)])
        baseline_df = create_pyspark_dataframe(float_data, schema)
        double_data = [(3.55,), (6.88,), (7.99,)]
        schema = StructType([
            StructField("target", DoubleType(), True)])
        production_df = create_pyspark_dataframe(double_data, schema)
        assert try_get_common_columns_with_error(baseline_df, production_df) == {'target': 'double'}

        # Test with two dataframes that have no common columns
        baseline_df = create_pyspark_dataframe([(1, "a"), (2, "b")],
                                               ["id", "name"])
        production_df = create_pyspark_dataframe([(3, "c"), (4, "d")],
                                                 ["age", "gender"])
        with pytest.raises(Exception) as ex:
            try_get_common_columns_with_error(baseline_df, production_df)
        assert "Found no common columns between input datasets." in str(ex.value)

        # Test with two dataframes that have different Types in common columns
        baseline_df = create_pyspark_dataframe([(1.0, "a", 10), (2.0, "b", 20)],
                                               ["id", "name", "age"])
        production_df = create_pyspark_dataframe([(1, "c", 30), (2, "d", 40)],
                                                 ["id", "name", "age"])
        assert try_get_common_columns_with_error(baseline_df, production_df) == {"name": "string", "age": "bigint"}

    def try_get_common_columns_ignore(self):
        """Test scenarios for common columns with ignore."""
        # Test with two empty dataframes
        spark = SparkSession.builder.appName("test").getOrCreate()
        emp_RDD = spark.sparkContext.emptyRDD()
        # Create empty schema
        columns = StructType([])

        # Create an empty RDD with empty schema
        baseline_df = create_pyspark_dataframe(emp_RDD, columns)
        production_df = create_pyspark_dataframe(emp_RDD, columns)
        assert try_get_common_columns(baseline_df, production_df) == {}

        # Test with two dataframes that have common columns but datatype is not in the same type
        float_data = [(3.55,), (6.88,), (7.99,)]
        schema = StructType([
            StructField("target", FloatType(), True)])
        baseline_df = create_pyspark_dataframe(float_data, schema)
        double_data = [(3.55,), (6.88,), (7.99,)]
        schema = StructType([
            StructField("target", DoubleType(), True)])
        production_df = create_pyspark_dataframe(double_data, schema)
        assert try_get_common_columns(baseline_df, production_df) == {'target': 'double'}

        # Test with two dataframes that have no common columns
        baseline_df = create_pyspark_dataframe([(1, "a"), (2, "b")],
                                               ["id", "name"])
        production_df = create_pyspark_dataframe([(3, "c"), (4, "d")],
                                                 ["age", "gender"])
        assert try_get_common_columns(baseline_df, production_df) == {}

        # Test with two dataframes that have different Types in common columns
        baseline_df = create_pyspark_dataframe([(1.0, "a", 10), (2.0, "b", 20)],
                                               ["id", "name", "age"])
        production_df = create_pyspark_dataframe([(1, "c", 30), (2, "d", 40)],
                                                 ["id", "name", "age"])
        assert try_get_common_columns(baseline_df, production_df) == {"name": "string", "age": "bigint"}

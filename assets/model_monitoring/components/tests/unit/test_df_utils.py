# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""This file contains unit tests for the df utilities."""

from pyspark.sql.types import (
    DoubleType,
    FloatType,
    StructField,
    StructType)
from pyspark.sql import SparkSession
from shared_utilities.df_utils import (
        get_numerical_cols_with_df,
        get_categorical_cols_with_df,
        is_categorical,
        is_numerical,
        get_common_columns,
        get_feature_type_override_map,
        is_numerical_new,
        is_categorical_new,
        get_numerical_cols_with_df_with_override,
        get_categorical_cols_with_df_with_override
    )
from tests.e2e.utils.io_utils import create_pyspark_dataframe
import pandas as pd
import pytest
import datetime


@pytest.mark.unit
class TestDFUtils:
    """Test class for df utilities."""

    def test_get_numerical_cols_with_df(self):
        """Test numerical columns."""
        # Test with mixed columns
        column_dtype_map = {'col1': 'int',
                            'col2': 'float',
                            'col3': 'double',
                            'col4': 'decimal',
                            'col5': 'string'
                            }
        baseline_df = pd.DataFrame({
                                    'col1': [1, 2, 2, 4, 4],
                                    'col2': [1.1, 2.2, 3.3, 4.4, 5.5],
                                    'col3': [1.11, 2.22, 3.33, 4.44, 5.55],
                                    'col4': [1.111, 2.222, 3.333, 4.444, 5.555]
                                    })
        baseline_df = self.init_spark().createDataFrame(baseline_df)
        numerical_columns = get_numerical_cols_with_df(column_dtype_map,
                                                       baseline_df)
        assert numerical_columns == ['col1', 'col2', 'col3', 'col4']

    def test_get_categorical_columns(self):
        """Test categorical columns."""
        # Test with all numerical columns
        spark = self.init_spark()
        column_dtype_map = {
                            'col1': 'int',
                            'col2': 'float',
                            'col3': 'double',
                            'col4': 'decimal'
                            }
        baseline_df = pd.DataFrame({
                                    'col1': [1, 2, 2, 4, 4],
                                    'col2': [1.1, 2.2, 3.3, 4.4, 5.5],
                                    'col3': [1.11, 2.22, 3.33, 4.44, 5.55],
                                    'col4': [1.111, 2.222, 3.333, 4.444, 5.555]
                                    })
        baseline_df = spark.createDataFrame(baseline_df)
        categorical_columns = get_categorical_cols_with_df(column_dtype_map,
                                                           baseline_df)
        assert categorical_columns == []

        # Test with int being same value
        column_dtype_map = {'col1': 'int'}
        baseline_df = pd.DataFrame({
                                    'col1': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                             1]
                                    })
        baseline_df = spark.createDataFrame(baseline_df)
        categorical_columns = get_categorical_cols_with_df(column_dtype_map,
                                                           baseline_df)
        assert categorical_columns == ['col1']

        # Test with all categorical columns
        column_dtype_map = {
            'col1': 'string',
            'col2': 'bool'
        }
        baseline_df = pd.DataFrame({
            'col1': ['a', 'b', 'c', 'd', 'e'],
            'col2': [True, False, True, False, True]
        })
        baseline_df = spark.createDataFrame(baseline_df)
        categorical_columns = get_categorical_cols_with_df(column_dtype_map,
                                                           baseline_df)
        assert categorical_columns == ['col1', 'col2']

    def test_is_categorical(self):
        """Test int is categorical."""
        # Test with integer column
        result = False
        baseline_column = pd.DataFrame([1, 2, 3, 4, 5])
        assert is_categorical(baseline_column) == result

        # Test with integer column with low distinct value ratio
        result = True
        baseline_column = pd.DataFrame([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        assert is_categorical(baseline_column) == result

        # Test with integer column with high distinct value ratio
        result = False
        baseline_column = pd.DataFrame([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        assert is_categorical(baseline_column) == result

    def test_is_numerical(self):
        """Test int is numerical."""
        # Test with integer column with high distinct value ratio
        baseline_column = pd.DataFrame([1, 2, 3, 4, 5])
        result = True
        assert is_numerical(baseline_column) == result

        # Test with integer column with low distinct value ratio
        result = False
        baseline_column = pd.DataFrame([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, None, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        assert is_numerical(baseline_column) == result

        # Test with integer column with high distinct value ratio
        result = True
        baseline_column = pd.DataFrame([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        assert is_numerical(baseline_column) == result

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
        # Test with no type override features
        override_numerical_features = None
        override_categorical_features = None
        assert get_feature_type_override_map(override_numerical_features, override_categorical_features) == {}

        # Test with only numerical override features
        override_numerical_features = "UserId,TransactionId"
        override_categorical_features = None
        assert get_feature_type_override_map(override_numerical_features, override_categorical_features) == {"UserId": "numerical",
                                                                                                            "TransactionId": "numerical"}

        # Test with only categorical override features
        override_numerical_features = None
        override_categorical_features = "Email"
        assert get_feature_type_override_map(override_numerical_features, override_categorical_features) == {"Email": "categorical"}

        # Test with both numerical and categorical override features
        override_numerical_features = "Age,Id"
        override_categorical_features = "Name,Gender"
        assert get_feature_type_override_map(override_numerical_features, override_categorical_features) == {"Age": "numerical",
                                                                                                             "Id": "numerical",
                                                                                                             "Name": "categorical",
                                                                                                             "Gender": "categorical"}

    def test_is_numerical_new(self):
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
        assert is_numerical_new('col2', column_dtype_map, feature_type_override_map, baseline_df) == False
        feature_type_override_map = {'col2': 'numerical'}
        assert is_numerical_new('col2', column_dtype_map, feature_type_override_map, baseline_df) == True

        # Test dtype_map take priority
        assert is_numerical_new('col2', column_dtype_map, {}, baseline_df) == True
        assert is_numerical_new('col3', column_dtype_map, {}, baseline_df) == True
        assert is_numerical_new('col4', column_dtype_map, {}, baseline_df) == True
        assert is_numerical_new('col5', column_dtype_map, {}, baseline_df) == False
        assert is_numerical_new('col6', column_dtype_map, {}, baseline_df) == False
        assert is_numerical_new('col7', column_dtype_map, {}, baseline_df) == False
        assert is_numerical_new('col8', column_dtype_map, {}, baseline_df) == False

        # Test with integer column with high distinct ratio
        baseline_df = pd.DataFrame({'col1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        baseline_df = spark.createDataFrame(baseline_df)
        assert is_numerical_new('col1', column_dtype_map, {}, baseline_df) == True

        # Test with integer column with low distinct value ratio
        baseline_df = pd.DataFrame({'col1': 
                                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, None, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]})
        baseline_df = spark.createDataFrame(baseline_df)                      
        assert is_numerical_new('col1', column_dtype_map, {}, baseline_df) == False

        # Test with unknown data type
        assert is_numerical_new('col9', column_dtype_map, {}, baseline_df) == False
        assert is_numerical_new('col10', column_dtype_map, {}, baseline_df) == False

    def test_is_categorical_new(self):
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
        assert is_categorical_new('col2', column_dtype_map, feature_type_override_map, baseline_df) == True
        feature_type_override_map = {'col2': 'numerical'}
        assert is_categorical_new('col2', column_dtype_map, feature_type_override_map, baseline_df) == False

        # Test dtype_map take priority
        assert is_categorical_new('col2', column_dtype_map, {}, baseline_df) == False
        assert is_categorical_new('col3', column_dtype_map, {}, baseline_df) == False
        assert is_categorical_new('col4', column_dtype_map, {}, baseline_df) == False
        assert is_categorical_new('col5', column_dtype_map, {}, baseline_df) == True
        assert is_categorical_new('col6', column_dtype_map, {}, baseline_df) == True
        assert is_categorical_new('col7', column_dtype_map, {}, baseline_df) == True
        assert is_categorical_new('col8', column_dtype_map, {}, baseline_df) == True

        # Test with integer column with high distinct ratio
        baseline_df = pd.DataFrame({'col1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        baseline_df = spark.createDataFrame(baseline_df)
        assert is_categorical_new('col1', column_dtype_map, {}, baseline_df) == False

        # Test with integer column with low distinct value ratio
        baseline_df = pd.DataFrame({'col1': 
                                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, None, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]})
        baseline_df = spark.createDataFrame(baseline_df)                      
        assert is_categorical_new('col1', column_dtype_map, {}, baseline_df) == True

        # Test with unknown data type
        assert is_categorical_new('col9', column_dtype_map, {}, baseline_df) == False
        assert is_categorical_new('col10', column_dtype_map, {}, baseline_df) == False

    def test_get_numerical_cols_with_df_with_override(self):
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

        # Test without dtype map without feature type override
        assert get_numerical_cols_with_df_with_override({}, baseline_df) == ['col1', 'col2', 'col3', 'col4']

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
                    'col9': 'date'
                    }
        assert get_numerical_cols_with_df_with_override({}, baseline_df, column_dtype_map) == ['col1', 'col2', 'col3', 'col4', 'col5']

        # Test with feature type override
        feature_type_override_map = {
                                    'col5': 'numerical',
                                    'col6': 'numerical',
                                    'col7': 'numerical',
                                    'col8': 'numerical',
                                    'col9': 'numerical'
                                    }
        assert get_numerical_cols_with_df_with_override(feature_type_override_map, baseline_df, column_dtype_map) == ['col1', 'col2', 'col3', 'col4', 'col5', 
                                                                                                                      'col6', 'col7', 'col8', 'col9']

    def test_get_categorical_cols_with_df_with_override(self):
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

        # Test without dtype map without feature type override
        assert get_categorical_cols_with_df_with_override({}, baseline_df) == ['col5', 'col6', 'col7', 'col8', 'col9']

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
                    'col9': 'date'
                    }
        assert get_categorical_cols_with_df_with_override({}, baseline_df, column_dtype_map) == ['col6', 'col7', 'col8', 'col9']

        # Test with feature type override
        feature_type_override_map = {
                                    'col1': 'categorical',
                                    'col2': 'categorical',
                                    'col3': 'categorical',
                                    'col4': 'categorical',
                                    'col5': 'categorical'
                                    }
        assert get_categorical_cols_with_df_with_override(feature_type_override_map, baseline_df, column_dtype_map) == ['col1', 'col2', 'col3', 'col4', 'col5', 
                                                                                                                        'col6', 'col7', 'col8', 'col9']

    def init_spark(self):
        """Get or create spark session."""
        spark = SparkSession.builder.appName("test").getOrCreate()
        return spark

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains the core logic for data drift compute metrics component."""

from pyspark.context import SparkContext
from pyspark.sql.functions import col
from pyspark.sql.session import SparkSession
from pyspark.sql.types import StructType, StructField, StringType
import pyspark
import pyspark.pandas as ps
from shared_utilities.df_utils import get_numerical_and_categorical_cols

# Init spark session
sc = SparkContext.getOrCreate()
spark = SparkSession(sc)


def get_df_schema(df: pyspark.sql.DataFrame) -> pyspark.sql.DataFrame:
    """
    Compute a Spark DataFrame containing the data type information and column names of the input Spark DataFrame.

    Args:
        df: Input Spark DataFrame.

    Returns:
        metadata_df: A Spark DataFrame containing the data type information and column names
    of the input Spark DataFrame.
    """
    # Get the schema of the DataFrame
    schema = df.schema

    # Define a new schema for the DataFrame that will hold the metadata
    metadata_schema = StructType(
        [
            StructField("featureName", StringType(), True),
            StructField("dataType", StringType(), True),
        ]
    )

    # Iterate through the columns of the schema and extract the metadata
    metadata_rows = []
    for col_ in schema:
        metadata_rows.append(
            (
                col_.name,
                str(col_.dataType),
            )
        )

    # Create a new DataFrame with the metadata
    metadata_df = spark.createDataFrame(metadata_rows, metadata_schema)

    return metadata_df


def get_features_for_max_min_calculation(df: pyspark.sql.DataFrame, numerical_columns: list) -> pyspark.sql.DataFrame:
    """
    Compute a Spark DataFrame with features which get max and min value.

    Args:
        df: Input Spark DataFrame.
        numerical_columns: list of numerical columns

    Returns:
        df_for_max_min_value: A Spark DataFrame with features which get max and min value
    """
    df_for_max_min_value = df.select(*numerical_columns)

    return df_for_max_min_value


def get_unique_value_list(df: pyspark.sql.DataFrame, categorical_columns: list) -> ps.DataFrame:
    """
    Get the unique values for each categorical column in a DataFrame.

    Args:
        df (pyspark.sql.DataFrame): Input DataFrame.
        categorical_columns: list of categorical columns

    Returns:
        pyspark.sql.DataFrame:  A Pypsark Pandas DataFrame containing the unique values for each categorical column.
        The DataFrame has two columns:
        featureName: The name of the categorical column.
        set: A list of unique values for the categorical column.
    """
    # Ignore bool, time, date categorical columns because they are meaningless for data quality calculation
    modified_categorical_columns = []
    dtype_map = dict(df.dtypes)
    for column in categorical_columns:
        if dtype_map[column] not in ["boolean", "timestamp", "date"]:
            modified_categorical_columns.append(column)

    # Compute the set of unique values for each categorical column
    unique_vals = [
        df.select(col(c)).distinct().rdd.map(lambda x: x[0]).collect()
        for c in modified_categorical_columns
    ]
    metadata_schema = StructType(
        [
            StructField("featureName", StringType(), True),
            StructField("set", StringType(), True),
        ]
    )

    # Create a new DataFrame with the results
    unique_vals_df = spark.createDataFrame(
        [(modified_categorical_columns[i], unique_vals[i]) for i in range(len(modified_categorical_columns))],
        schema=metadata_schema,
    )

    unique_vals_df = unique_vals_df.to_pandas_on_spark()
    return unique_vals_df


def compute_min_df(df: ps.DataFrame) -> ps.DataFrame:
    """
    Compute the minimum value for each feature in the input DataFrame.

    Args:
        df: A pandas-on-Spark DataFrame.

    Returns:
        A pandas-on-Spark DataFrame containing two columns: "featureName" and "min_value".
        The "featureName" column lists the name of each feature in the input DataFrame, and
        the "min_value" column lists the minimum value for each feature.
    """
    min_vals = ps.DataFrame(data=df.min(axis=0)).reset_index()
    min_vals.columns = ["featureName", "min_value"]

    return min_vals


def compute_max_df(df: ps.DataFrame) -> ps.DataFrame:
    """
    Compute the maximum value for each feature in the input DataFrame.

    Args:
        df: A pandas-on-Spark DataFrame.

    Returns:
        A pandas-on-Spark DataFrame containing two columns: "featureName" and "max_value".
        The "featureName" column lists the name of each feature in the input DataFrame, and
        the "max_value" column lists the maximum value for each feature.
    """
    max_vals = ps.DataFrame(data=df.max(axis=0)).reset_index()
    max_vals.columns = ["featureName", "max_value"]

    return max_vals


def compute_data_quality_statistics(df, override_numerical_features, override_categorical_features) -> ps.DataFrame:
    """Compute data quality statistics."""
    dtype_df = get_df_schema(df=df).to_pandas_on_spark()
    numerical_columns, categorical_columns = get_numerical_and_categorical_cols(
                                                            df,
                                                            override_numerical_features,
                                                            override_categorical_features)

    unique_vals_df = get_unique_value_list(df=df, categorical_columns=categorical_columns)
    # for max_vals and min_vals.
    df_for_max_min_value = get_features_for_max_min_calculation(df=df, numerical_columns=numerical_columns)
    # The compute_max_df and compute_min_df works for all numerical, except ShortType()
    # They will get null for non-numerical data
    max_vals = compute_max_df(df=df_for_max_min_value.to_pandas_on_spark())
    min_vals = compute_min_df(df=df_for_max_min_value.to_pandas_on_spark())

    # Join tables to get all metrics into one table
    min_max_df = max_vals.merge(min_vals, left_on="featureName", right_on="featureName")
    metric_df = min_max_df.merge(
        dtype_df, right_on="featureName", left_on="featureName", how="right"
    )
    metric_unique_df = metric_df.merge(
        unique_vals_df, right_on="featureName", left_on="featureName", how="left"
    )
    return metric_unique_df

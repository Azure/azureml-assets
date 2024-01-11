# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains the core logic for data drift compute metrics component."""

from pyspark.context import SparkContext
from pyspark.sql.functions import col
from pyspark.sql.session import SparkSession
from pyspark.sql.types import DoubleType, LongType, StructType, StructField, StringType
import pyspark
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


def get_unique_value_list(df: pyspark.sql.DataFrame, categorical_columns: list) -> pyspark.sql.DataFrame:
    """
    Get the unique values for each categorical column in a DataFrame.

    Args:
        df (pyspark.sql.DataFrame): Input DataFrame.
        categorical_columns: list of categorical columns

    Returns:
        pyspark.sql.DataFrame:  A Pypsark DataFrame containing the unique values for each categorical column.
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

    return unique_vals_df


def compute_max_and_min_df(df: pyspark.sql.DataFrame, dtype_df: pyspark.sql.DataFrame) -> pyspark.sql.DataFrame:
    """
    Compute the maximum value for each feature in the input DataFrame.

    Args:
        df: A spark data frame with features to calculate max value and min value。
        dtype_df: A spark data frame for datatype for each featureName

    Returns:
        A Spark DataFrame containing two columns: "featureName" and "max_value" amd "min_value"
        The "featureName" column lists the name of each feature in the input DataFrame, and
        The "max_value" column lists the maximum value for each feature.
        The "min_value" column lists the minimum value for each feature.
    """
    # check if the data type contains double or float
    # we use the higher precision for the max and min value
    # If datatype does not contain double or float, then we use LongType for all other datatype
    struct_fields = [StructField("featureName", StringType(), True)]
    is_double_type = True
    if bool(dtype_df.filter(dtype_df.dataType.contains("DoubleType")).collect())\
       or bool(dtype_df.filter(dtype_df.dataType.contains("FloatType")).collect()):
        struct_fields.append(StructField("max_value", DoubleType(), True))
        struct_fields.append(StructField("min_value", DoubleType(), True))
    else:
        struct_fields.append(StructField("max_value", LongType(), True))
        struct_fields.append(StructField("min_value", LongType(), True))
        is_double_type = False

    data_schema = StructType(struct_fields)

    max_and_min_value_rows = []
    for row in dtype_df.collect():
        if row["featureName"] in df.columns:
            if is_double_type:
                max_and_min_value_rows.append(
                    (row["featureName"],
                     float(df.agg({row["featureName"]: "max"}).collect()[0][0]),
                     float(df.agg({row["featureName"]: "min"}).collect()[0][0]))
                    )
            else:
                max_and_min_value_rows.append(
                    (row["featureName"],
                     df.agg({row["featureName"]: "max"}).collect()[0][0],
                     df.agg({row["featureName"]: "min"}).collect()[0][0])
                    )

    max_and_min_vals_df = spark.createDataFrame(
        max_and_min_value_rows,
        schema=data_schema,
    )

    return max_and_min_vals_df


def compute_data_quality_statistics(df, override_numerical_features, override_categorical_features) -> pyspark.sql.DataFrame:
    """Compute data quality statistics."""
    dtype_df = get_df_schema(df=df)
    unique_vals_df = get_unique_value_list(df=df, categorical_columns=categorical_columns)
    # Note: excluding boolean type column as the boolean type do not need to be calculated
    # for max_vals and min_vals.
    # They will get null for non-numerical data
    df_for_max_min_value = get_features_for_max_min_calculation(df=df, numerical_columns=numerical_columns)

    # Join tables to get all metrics into one table
    min_max_df = compute_max_and_min_df(df_for_max_min_value, dtype_df)
    metric_df = min_max_df.join(
        dtype_df, ["featureName"], "right"
    )
    metric_unique_df = metric_df.join(
        unique_vals_df, ["featureName"], "left"
    )
    return metric_unique_df

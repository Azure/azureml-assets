# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains the core logic for data drift compute metrics component."""

from pyspark.context import SparkContext
from pyspark.sql.functions import col
from pyspark.sql.session import SparkSession
from pyspark.sql.types import DoubleType, LongType, StructType, StructField, StringType
import pyspark
import pyspark.pandas as ps


# Init spark session
sc = SparkContext.getOrCreate()
spark = SparkSession(sc)

supported_datatype_for_max_min_value = ["IntegerType()", "DoubleType()", "ByteType()",
                                        "LongType()", "FloatType()", "ShortType()"]


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


def get_features_for_max_min_calculation(df: pyspark.sql.DataFrame) -> pyspark.sql.DataFrame:
    """
    Compute a Spark DataFrame with features which get max and min value.

    Args:
        df: Input Spark DataFrame.

    Returns:
        df_for_max_min_value: A Spark DataFrame with features which get max and min value
    """
    schema = df.schema
    supported_columns = []
    for col_ in schema:
        if str(col_.dataType) in supported_datatype_for_max_min_value:
            supported_columns.append(col_.name)

    df_for_max_min_value = df.select(*supported_columns)

    return df_for_max_min_value


def get_unique_value_list(df: pyspark.sql.DataFrame) -> pyspark.sql.DataFrame:
    """
    Get the unique values for each categorical column in a DataFrame.

    Args:
        df (pyspark.sql.DataFrame): Input DataFrame.

    Returns:
        pyspark.sql.DataFrame:  A Pypsark DataFrame containing the unique values for each categorical column.
        The DataFrame has two columns:
        featureName: The name of the categorical column.
        set: A list of unique values for the categorical column.
    """
    # Define a list of categorical column names to process
    cat_col_names = [c[0] for c in df.dtypes if c[1] == "string"]

    # Compute the set of unique values for each categorical column
    unique_vals = [
        df.select(col(c)).distinct().rdd.map(lambda x: x[0]).collect()
        for c in cat_col_names
    ]
    metadata_schema = StructType(
        [
            StructField("featureName", StringType(), True),
            StructField("set", StringType(), True),
        ]
    )

    # Create a new DataFrame with the results
    unique_vals_df = spark.createDataFrame(
        [(cat_col_names[i], unique_vals[i]) for i in range(len(cat_col_names))],
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
        A Spark DataFrame containing two columns: "featureName" and "max_value".
        The "featureName" column lists the name of each feature in the input DataFrame, and
        the "max_value" column lists the maximum value for each feature.
    """
    # check if the data type contains double or float
    # we use the higher precision for the max and min value
    # If datatype does not contain double or float, then we use LongType for all other datatype
    struct_fields = [StructField("featureName", StringType(), True)]
    is_double_type = True
    if bool(dtype_df.filter(dtype_df.dataType.contains("DoubleType")).collect())\
    or bool(dtype_df.filter(dtype_df.dataType.contains("FloatType")).collect()):
        struct_fields.append(StructField("min_value", DoubleType(), True))
        struct_fields.append(StructField("max_value", DoubleType(), True))
    else:
        struct_fields.append(StructField("min_value", LongType(), True))
        struct_fields.append(StructField("max_value", LongType(), True))
        is_double_type = False

    data_schema = StructType(struct_fields)

    max_and_min_value_rows = []
    for row in dtype_df.collect():
        if row["featureName"] in df.columns:
            if is_double_type:
                max_and_min_value_rows.append(
                    (row["featureName"],
                     float(df.agg({row["featureName"]: "min"}).collect()[0][0]),
                     float(df.agg({row["featureName"]: "max"}).collect()[0][0]))
                    )
            else:
                max_and_min_value_rows.append(
                    (row["featureName"],
                     df.agg({row["featureName"]: "min"}).collect()[0][0],
                     df.agg({row["featureName"]: "max"}).collect()[0][0])
                    )

    max_and_min_vals_df = spark.createDataFrame(
        max_and_min_value_rows,
        schema=data_schema,
    )

    return max_and_min_vals_df


def compute_data_quality_statistics(df) -> pyspark.sql.DataFrame:
    """Compute data quality statistics."""
    dtype_df = get_df_schema(df=df)
    unique_vals_df = get_unique_value_list(df=df)
    # Note: excluding boolean type column as the boolean type do not need to be calculated
    # for max_vals and min_vals.
    # They will get null for non-numerical data
    df_for_max_min_value = get_features_for_max_min_calculation(df=df)

    # Join tables to get all metrics into one table
    min_max_df = compute_max_and_min_df(df_for_max_min_value, dtype_df)
    metric_df = min_max_df.join(
        dtype_df, min_max_df.featureName == dtype_df.featureName, "right"
    )
    metric_unique_df = metric_df.join(
        unique_vals_df, metric_df.featureName == unique_vals_df.featureName, "left"
    )
    return metric_unique_df

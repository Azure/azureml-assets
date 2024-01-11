# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains the core logic for data quality metrics component."""

from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    ByteType,
    ShortType,
    IntegerType,
    LongType,
    FloatType,
    DoubleType,
    DecimalType,
)
from pyspark.sql.functions import (
    col,
    lit,
    count,
    when,
    desc,
    regexp_replace,
    round,
    concat,
    isnan
)
from pyspark.ml.feature import Imputer
from typing import Tuple
import pyspark
import warnings


# Init spark session
sc = SparkContext.getOrCreate()
spark = SparkSession(sc)
supported_datatypes = ['timestamp', 'boolean', 'double', 'binary', 'date', 'float', 'integer',
                       'long', 'short', 'string', 'byte']


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


def get_null_count(df: pyspark.sql.DataFrame) -> pyspark.sql.DataFrame:
    """
    Compute a PySpark DataFrame containing the number of null values for each column of the input PySpark DataFrame.

    Args:
        df: Input PySpark DataFrame.

    Returns:
        na_metric_df: A Pypsark DataFrame containing the number of null values
    for each column of the input PySpark DataFrame.
    """
    na_metric_df = df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df.columns])

    na_metric_df_data = [(col_, na_metric_df.first()[col_]) for col_ in na_metric_df.columns]
    data_schema = StructType(
        [
            StructField("featureName", StringType(), True),
            StructField("violationCount", IntegerType(), True),
        ]
    )
    # Create a new DataFrame with the metric data
    na_metric_df = spark.createDataFrame(na_metric_df_data, data_schema)
    return na_metric_df


def compute_max_violation(
    df: pyspark.sql.DataFrame, data_stats_table: pyspark.sql.DataFrame
) -> pyspark.sql.DataFrame:
    """
    Compute the maximum threshold violation count for numerical columns in the input PySpark DataFrame.

    Args:
        df: Input PySpark DataFrame. (baseline or, target dataset)
        data_stats_table: Input data statistics table. PySpark DataFrame.

    Returns:
        max_violation_df: A PySpark DataFrame with columns for violation count, feature name and metric name.
    """
    max_threshold_violation_count = []
    feature_name_list = []

    numerical_types = ["double", "float", "int", "bigint", "smallint", "tinyint", "long"]

    for row in data_stats_table.filter(col("min_value").isNotNull()).select("featureName").distinct().collect():
        feature_name = row["featureName"]

        if feature_name not in df.columns:
            continue

        if df.select(feature_name).dtypes[0][1] in numerical_types:
            feature_name_list.append(feature_name)
            data_stats_table_subset = data_stats_table.filter(
                col("featureName") == feature_name
            )
            df_subset = df.select(feature_name)
            max_threshold_violation_count.append(
                df_subset.filter(
                    col(feature_name)
                    > data_stats_table_subset.select("max_value").collect()[0][
                        "max_value"
                    ]
                ).count()
            )
        else:
            feature_name_list.append(feature_name)
            max_threshold_violation_count.append(None)
    max_violation_df = spark.createDataFrame(
        zip(max_threshold_violation_count, feature_name_list),
        StructType(
            [
                StructField("violationCount", IntegerType(), True),
                StructField("featureName", StringType(), True),
            ]
        ),
    )
    max_violation_df = max_violation_df.withColumn(
        "metricName", lit("maxValueOutOfRange")
    )

    return max_violation_df


def compute_min_violation(
    df: pyspark.sql.DataFrame, data_stats_table: pyspark.sql.DataFrame
) -> pyspark.sql.DataFrame:
    """
    Compute the minimum threshold violation count for numerical columns in the input PySpark DataFrame.

    Args:
        df: Input PySpark DataFrame.
        data_stats_table: Input data statistics table (also a PySpark DataFrame).

    Returns:
        min_violation_df: A PySpark DataFrame with columns for violation count, feature name and metric name.
    """
    min_threshold_violation_count = []
    feature_name_list = []

    numerical_types = ["double", "float", "int", "bigint", "smallint", "tinyint", "long"]

    for row in data_stats_table.filter(col("max_value").isNotNull()).select("featureName").distinct().collect():
        feature_name = row["featureName"]

        if feature_name not in df.columns:
            continue

        if df.select(feature_name).dtypes[0][1] in numerical_types:
            feature_name_list.append(feature_name)
            data_stats_table_subset = data_stats_table.filter(
                col("featureName") == feature_name
            )
            df_subset = df.select(feature_name)
            min_threshold_violation_count.append(
                df_subset.filter(
                    col(feature_name)
                    < data_stats_table_subset.select("min_value").collect()[0][
                        "min_value"
                    ]
                ).count()
            )
        else:
            feature_name_list.append(feature_name)
            min_threshold_violation_count.append(None)

    min_violation_df = spark.createDataFrame(
        zip(min_threshold_violation_count, feature_name_list),
        StructType(
            [
                StructField("violationCount", IntegerType(), True),
                StructField("featureName", StringType(), True),
            ]
        ),
    )
    min_violation_df = min_violation_df.withColumn(
        "metricName", lit("minValueOutOfRange")
    )
    return min_violation_df


def compute_set_violation(
    df: pyspark.sql.DataFrame, data_stats_table: pyspark.sql.DataFrame
) -> pyspark.sql.DataFrame:
    """
    Compute the count of values in a column that are not in the allowed set of values specified.

    Args:
        df: A PySpark Pandas DataFrame containing the data to check.
        data_stats_table: A PySpark DataFrame containing metadata about the data,
        including the allowed set of values for each column

    Returns:
        threshold_violation_df: A PySpark DataFrame with the count of values in each column
        that are not in the allowed set of values
    """
    set_threshold_violation_count = []
    feature_name_list = []

    for c in (
        data_stats_table
        .filter(col("set").isNotNull())
        .select("featureName")
        .distinct()
        .rdd.flatMap(lambda x: x)
        .collect()
    ):
        if c not in df.columns:
            continue

        df_subset = data_stats_table.filter(col("featureName") == c)
        if df_subset.filter(col("dataType") == "StringType").count() >= 1:
            set_subset = df_subset.select("set").rdd.flatMap(lambda x: x).first()
            set_threshold_violation_count.append(
                df.filter(~col(c).isin(set_subset)).count()
            )
            feature_name_list.append(c)
        else:
            set_threshold_violation_count.append(0)
            feature_name_list.append(c)

    threshold_violation_df = spark.createDataFrame(
        zip(feature_name_list, set_threshold_violation_count),
        schema=StructType(
            [
                StructField("featureName", StringType(), True),
                StructField("violationCount", IntegerType(), True),
            ]
        ),
    )
    threshold_violation_df = threshold_violation_df.withColumn(
        "metricName", lit("setValueOutOfRange")
    )
    return threshold_violation_df


def compute_dtype_violation_count_modify_dataset(
    df: pyspark.sql.DataFrame, data_stats_table_mod: pyspark.sql.DataFrame
) -> Tuple[pyspark.sql.DataFrame, pyspark.sql.DataFrame]:
    """
    Compute the number of data type violations for each column in the input DataFrame.

    Args:
        df (pyspark.sql.DataFrame): Input DataFrame
        data_stats_table_mod (pyspark.sql.DataFrame): Data statistics table DataFrame with expected data types

    Returns:
        Tuple[pyspark.sql.DataFrame, pyspark.sql.DataFrame]: A tuple of two DataFrames:
        1. df: The modified input DataFrame with the data types of columns with errors cast to their expected
        data type from the data_stats_table_mod
        2. df_conversion_errors: A DataFrame with the count of data type violations for each column in the
        input DataFrame and the corresponding metric name
    """
    column_names = "featureName|violationCount"
    mySchema = StructType(
        [StructField(c, StringType()) for c in column_names.split("|")]
    )
    df_conversion_errors = spark.createDataFrame([], schema=mySchema)

    # Loop through each column in the original DataFrame
    for column in df.columns:

        dtype_baseline = data_stats_table_mod.select(["featureName", "dataType"]) \
            .filter(data_stats_table_mod.featureName == column) \
            .select("dataType")\
            .collect()

        if not dtype_baseline:
            print(f"Feature '{column}' is not present in data statistics. " +
                  f"Skipping data type violation count for '{column}'.")
            continue

        dtype_baseline = dtype_baseline[0][0]

        if dtype_baseline not in supported_datatypes:
            # we do not support this type validation, we do not count num_errors for unsupported datatype
            warnings.warn("we do not support this datatype {} validation".format(dtype_baseline))
            continue
        else:
            # Cast the column to datatype from baseline reference column and count the number of errors
            num_errors = (
                df.select(column)
                .where(~col(column).cast(dtype_baseline).isNotNull())
                .count()
            )
        # Add the conversion error count to the new DataFrame
        df_conversion_errors = df_conversion_errors.union(
            spark.createDataFrame([(column, num_errors)], schema=mySchema)
        )

        # Change the data type to dtype_baseline for the affected column
        if num_errors > 0:
            df = df.withColumn(column, col(column).cast(dtype_baseline))

    df_conversion_errors = df_conversion_errors.withColumn(
        "violationCount", col("violationCount").cast(IntegerType())
    )
    df_conversion_errors = df_conversion_errors.withColumn(
        "metricName", lit("DataTypeError")
    )

    # Show the new DataFrame and df_conversion_errors
    return df, df_conversion_errors


def impute_numericals_with_median(df: pyspark.sql.DataFrame) -> pyspark.sql.DataFrame:
    """
    Impute missing values in numerical columns with the median value of that column.

    Args:
    df (pyspark.sql.DataFrame): The input DataFrame

    Returns:
    df (pyspark.sql.DataFrame): The input DataFrame with missing values in numerical columns imputed with median
    """
    dbl_cols = [f.name for f in df.schema.fields if isinstance(f.dataType, (ByteType, ShortType, IntegerType, LongType, FloatType, DoubleType, DecimalType))] # noqa

    if len(dbl_cols) > 0:
        imputer = Imputer(inputCols=dbl_cols, outputCols=[c for c in dbl_cols]).setStrategy(
            "median"
        )
        # Fit imputer on Data Frame and Transform it
        df = imputer.fit(df).transform(df)
    return df


def impute_categorical_with_mode(df: pyspark.sql.DataFrame) -> pyspark.sql.DataFrame:
    """
    Impute missing values in numerical columns with the mode/most frequent value of that column.

    Args:
    df (pyspark.sql.DataFrame): The input DataFrame

    Returns:
    df (pyspark.sql.DataFrame): The input DataFrame with missing values
    in numerical columns imputed with mode/most frequent
    """
    categorical_col = [
        f.name for f in df.schema.fields if isinstance(f.dataType, StringType)
    ]

    for i in categorical_col:
        # Find the most frequent value in the column
        # Get the most frequent value in the categorical column
        most_frequent = (
            df.groupBy(i)
            .agg(count("*").alias("count"))
            .orderBy(desc("count"))
            .first()[i]
        )

        # Impute the missing values with the most frequent value
        df = df.fillna({i: most_frequent})

    return df


def modify_dataType(data_stats_table) -> pyspark.sql.DataFrame:
    """Cast DataType() to DataType."""
    # When adding new datatype into the cast list, please also add it into supported_datatypes list
    data_stats_table_mod = data_stats_table.withColumn(
        "dataType",
        when(col("dataType") == "DoubleType()", "double")
        .when(col("dataType") == "StringType()", "string")
        .when(col("dataType") == "IntegerType()", "integer")
        .when(col("dataType") == "LongType()", "long")
        .when(col("dataType") == "TimestampType()", "timestamp")
        .when(col("dataType") == "BooleanType()", "boolean")
        .when(col("dataType") == "BinaryType()", "binary")
        .when(col("dataType") == "DateType()", "date")
        .when(col("dataType") == "FloatType()", "float")
        .when(col("dataType") == "ShortType()", "short")
        .when(col("dataType") == "ByteType()", "byte")
        .otherwise(col("dataType")),
    )
    return data_stats_table_mod


def compute_data_quality_metrics(df, data_stats_table):
    """Compute data quality metrics."""
    #########################
    # PREPARE THE DATA
    #########################
    # Cache the DataFrames
    df.cache()
    data_stats_table.cache()

    data_stats_table_mod = modify_dataType(data_stats_table)

    #########################
    # COMPUTE VIOLATIONS
    #########################
    # 1. NULL TYPE
    null_count_dtype = get_null_count(df)
    null_count_dtype["metricName"] = "NullValue"
    null_count_dtype_sp = null_count_dtype.to_spark()
    # HIERARCHY 1: IMPUTE MISSING VALUES AFTER COUNTING THEM
    df = impute_numericals_with_median(df)
    df = impute_categorical_with_mode(df)

    # 2. DATA TYPE VIOLATION
    df, dtype_violation_df = compute_dtype_violation_count_modify_dataset(
        df=df, data_stats_table_mod=data_stats_table_mod
    )
    # HIERARCHY 2: CHANGE D-TYPE OF AFFECTED COLUMN TO BASELINE SCHEMA
    # THIS HAPPENS IN THE `compute_dtype_violation_count_modify_dataset` FUNCTION with df being overwritten

    # 3. OUT OF BOUNDS
    max_violation_df = compute_max_violation(df=df, data_stats_table=data_stats_table)
    min_violation_df = compute_min_violation(df=df, data_stats_table=data_stats_table)
    threshold_violation_df = compute_set_violation(
        df=df, data_stats_table=data_stats_table
    )
    data_stats_table.unpersist()
    data_stats_table_mod.unpersist()

    #########################
    # JOIN ALL TABLES
    #########################
    violation_df = max_violation_df.unionByName(min_violation_df)
    min_violation_df.unpersist()  # release pre-join data frames from memory
    max_violation_df.unpersist()

    temp_select = null_count_dtype_sp.select(
        ["featureName", "violationCount", "metricName"]
    )
    violation_df = temp_select.unionByName(violation_df)
    temp_select.unpersist()

    violation_df = violation_df.unionByName(threshold_violation_df)
    threshold_violation_df.unpersist()

    violation_df = violation_df.unionByName(dtype_violation_df)
    dtype_violation_df.unpersist()

    dtype_df = get_df_schema(df)
    violation_df = dtype_df.join(violation_df, ["featureName"], how="right")
    dtype_df.unpersist()

    # ADD ROW COUNT
    df_length = (
        spark.createDataFrame(
            [(df.count(), "RowCount")],
            schema=StructType(
                [
                    StructField("violationCount", IntegerType(), True),
                    StructField("metricName", StringType(), True),
                ]
            ),
        )
        .withColumn("featureName", lit(""))
        .withColumn("dataType", lit(""))
    )
    # add the new row to the original DataFrame using unionByName()
    violation_df = violation_df.unionByName(df_length)

    violation_df_remapped = violation_df.withColumn(
        "metricName",
        when(
            violation_df.metricName.endswith("maxValueOutOfRange"),
            regexp_replace(
                violation_df.metricName, "maxValueOutOfRange", "OutOfBounds"
            ),
        )
        .when(
            violation_df.metricName.endswith("minValueOutOfRange"),
            regexp_replace(
                violation_df.metricName, "minValueOutOfRange", "OutOfBounds"
            ),
        )
        .when(
            violation_df.metricName.endswith("setValueOutOfRange"),
            regexp_replace(
                violation_df.metricName, "setValueOutOfRange", "OutOfBounds"
            ),
        )
        .otherwise(violation_df.metricName),
    )

    violation_df_remapped = (
        violation_df_remapped.select(
            ["featureName", "metricName", "violationCount", "dataType"]
        )
        .groupby(["featureName", "metricName", "dataType"])
        .sum()
        .withColumnRenamed("sum(violationCount)", "violationCount")
    )

    # COMPUTE RATIOS
    # 'len_df' is the name of the column to divide which is the row count needed for the ratios
    len_df = (
        violation_df_remapped.filter(col("metricName") == "RowCount")
        .select("violationCount")
        .collect()[0][0]
    )
    # divide the column by the row count using
    violation_df_remapped = violation_df_remapped.withColumn(
        "metricValue", round(violation_df_remapped["violationCount"] / lit(len_df), 5)
    )

    # REMAP THE DATA TYPES
    # TODO: Expand datatype list below. Also NOTE this list does not match list in data drift.
    violation_df_remapped = violation_df_remapped.withColumn(
        "dataType",
        when(col("dataType") == "DoubleType()", "Numerical")
        .when(col("dataType") == "LongType()", "Numerical")
        .when(col("dataType") == "StringType()", "Categorical")
        .when(col("dataType") == "IntegerType()", "Numerical")
        .when(col("dataType") == "TimestampType()", "Categorical")
        .when(col("dataType") == "BooleanType()", "Categorical")
        .when(col("dataType") == "BinaryType()", "Categorical")
        .when(col("dataType") == "DateType()", "Categorical")
        .when(col("dataType") == "FloatType()", "Numerical")
        .when(col("dataType") == "ShortType()", "Numerical")
        .when(col("dataType") == "ByteType()", "Numerical")
        .otherwise(col("dataType")),
    )

    #########################
    # ALIGN COLUMN NAMING
    #########################
    # RENAME METRIC VALUE
    violation_df_remapped = violation_df_remapped.withColumn(
        "metricName",
        when(
            col("metricName") != "RowCount", concat(col("metricName"), lit("Rate"))
        ).otherwise(col("metricName")),
    )

    # MOVE ROW COUNT "metricValue" TO THE RIGHT COLUMN AND SET VIOLATION COUNT TO 0
    violation_df_remapped = violation_df_remapped.withColumn(
        "metricValue",
        when(col("metricName") == "RowCount", col("violationCount")).otherwise(
            col("metricValue")
        ),
    ).withColumn(
        "violationCount",
        when(col("metricName") == "RowCount", 0).otherwise(col("violationCount")),
    )

    return violation_df_remapped

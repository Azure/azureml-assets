# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains the core logic for the compute histogram component."""


import pyspark.sql as pyspark_sql
import pyspark.sql.functions as F
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    FloatType,
    IntegerType,
)
from shared_utilities.constants import (
    CATEGORICAL_FEATURE_CATEGORY,
    NUMERICAL_FEATURE_CATEGORY,
)
from shared_utilities.df_utils import get_categorical_cols_with_df, get_numerical_cols_with_df
from shared_utilities.histogram_utils import get_histograms
from shared_utilities.io_utils import init_spark


def generate_numerical_histogram_rows(histogram_dict: dict) -> list:
    """Generate row in histogram df with lower bound, upper bound, count per bin for a column."""
    entries = []
    for column_name in histogram_dict:
        col_edges = histogram_dict[column_name][0]
        col_bucket_counts = histogram_dict[column_name][1]
        for i in range(0, len(col_edges) - 1):
            lower_bound = round(col_edges[i], 2)
            upper_bound = round(col_edges[i + 1], 2)
            bucket_count = col_bucket_counts[i]
            # entry: [feature_name, bucket_count, lower_bound, upper_bound, categorical_bucket, datatype]
            entry = [
                column_name,
                bucket_count,
                float(lower_bound),
                float(upper_bound),
                "",
                NUMERICAL_FEATURE_CATEGORY,
            ]
            entries.append(entry)
    return entries


def generate_categorical_histogram_rows(df, column_names: list) -> list:
    """Generate row in histogram df with count per value for a column."""
    entries = []
    for column_name in column_names:
        count_df = df.groupBy(column_name).count()
        print(f"{column_name} count(): {count_df.count()}")
        VISUALIZATION_MAX_CATEGORICAL_BUCKETS = 100
        if count_df.count() > VISUALIZATION_MAX_CATEGORICAL_BUCKETS:
            print(
                f"WARNING: {column_name} has {count_df.count()} unique values."
                + "Only the top {VISUALIZATION_MAX_CATEGORICAL_BUCKETS} values will be visualized."
            )
            count_df = count_df.orderBy(F.desc("count")).limit(
                VISUALIZATION_MAX_CATEGORICAL_BUCKETS
            )
            for count_df_row in count_df.collect():
                # entry: [feature_name, bucket_count, lower_bound, upper_bound, categorical_bucket, datatype]
                entry = [
                    column_name,
                    count_df_row["count"],
                    float("nan"),
                    float("nan"),
                    count_df_row[column_name],
                    CATEGORICAL_FEATURE_CATEGORY,
                ]
                entries.append(entry)
            entries.append(
                [
                    column_name,
                    0,
                    float("nan"),
                    float("nan"),
                    "Other",
                    CATEGORICAL_FEATURE_CATEGORY,
                ]
            )
        else:
            for count_df_row in count_df.collect():
                # entry: [feature_name, bucket_count, lower_bound, upper_bound, categorical_bucket, datatype]
                entry = [
                    column_name,
                    count_df_row["count"],
                    float("nan"),
                    float("nan"),
                    count_df_row[column_name],
                    CATEGORICAL_FEATURE_CATEGORY,
                ]
                entries.append(entry)
    return entries


def get_histogram_spark_df_schema() -> StructType:
    """Get Histogram Spark DataFrame Schema."""
    schema = StructType(
        [
            StructField("feature_bucket", StringType(), True),
            StructField("bucket_count", IntegerType(), True),
            StructField("lower_bound", FloatType(), True),
            StructField("upper_bound", FloatType(), True),
            StructField("category_bucket", StringType(), True),
            StructField("data_type", StringType(), True),
        ]
    )
    return schema


def create_histogram_df(rows: list) -> pyspark_sql.DataFrame:
    """Create Output Spark DataFrame."""
    output_schema = get_histogram_spark_df_schema()
    spark = init_spark()
    return spark.createDataFrame(data=rows, schema=output_schema)


def _to_bin_edges(histogram_buckets: pyspark_sql.DataFrame):
    """Convert histogram buckets to bin edges."""
    bin_edges = {}
    for row in histogram_buckets.collect():
        feature_name = row["feature_name"]
        if feature_name not in bin_edges:
            bin_edges[feature_name] = []
        bin_edges[feature_name].append(row["bucket"])
    return bin_edges


def compute_histograms(
    df: pyspark_sql.DataFrame, histogram_buckets: pyspark_sql.DataFrame
) -> tuple:
    """Compute data drift measures and perform tests."""
    # Generate histograms only for columns in both baseline and target dataset
    columns_dict = {}
    df_dtypes = dict(df.dtypes)
    for (column_name, data_type) in df_dtypes.items():
        columns_dict[column_name] = data_type

    numerical_columns = get_numerical_cols_with_df(columns_dict,
                                                   df)
    categorical_columns = get_categorical_cols_with_df(columns_dict,
                                                       df)

    # Numerical column histogram generation

    bin_edges = _to_bin_edges(histogram_buckets)
    histogram_dict = get_histograms(df, bin_edges, numerical_columns)

    numerical_histogram_rows = generate_numerical_histogram_rows(
        histogram_dict)

    baseline_numerical_histogram_df = create_histogram_df(
        numerical_histogram_rows)

    # Categorical column histogram generation
    categorical_histogram_rows = generate_categorical_histogram_rows(
        df, categorical_columns
    )

    categorical_histogram_df = create_histogram_df(categorical_histogram_rows)

    # Generate baseline and production histogram with numerical and
    # categorical columns
    histogram_df = baseline_numerical_histogram_df.union(
        categorical_histogram_df)

    return histogram_df

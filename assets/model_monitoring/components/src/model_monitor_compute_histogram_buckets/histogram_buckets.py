# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains the core logic for compute histogram buckets component."""


import pyspark.sql as pyspark_sql
from assets.model_monitoring.components.src.shared_utilities.momo_exceptions import InvalidInputError
from shared_utilities.df_utils import get_numerical_cols_with_df
from shared_utilities.histogram_utils import get_dual_histogram_bin_edges
from shared_utilities.df_utils import get_common_columns
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    DoubleType,
)
from shared_utilities.io_utils import init_spark


def compute_numerical_bins(
    df1: pyspark_sql.DataFrame, df2: pyspark_sql.DataFrame
) -> tuple:
    """Compute numerical bins given two data frames."""
    # Generate histograms only for columns in both baseline and target dataset
    common_columns_dict = get_common_columns(df1, df2)
    if not common_columns_dict:
        raise InvalidInputError(
            "Found no common columns between input datasets. Try double-checking" +
            " if there are common columns between the input datasets." +
            " Common columns must have the same names (case-sensitive) and similar data types."
        )
    numerical_columns = get_numerical_cols_with_df(common_columns_dict, df1)
    print(f"numerical columns: {numerical_columns}")

    # Numerical column histogram generation
    baseline_count = df1.count()
    production_count = df2.count()

    bin_edges = get_dual_histogram_bin_edges(
        df1, df2, baseline_count, production_count, numerical_columns
    )
    return bin_edges


def compute_histogram_buckets(
    df1: pyspark_sql.DataFrame, df2: pyspark_sql.DataFrame
) -> pyspark_sql.DataFrame:
    """Compute histogram buckets."""
    print("compute numerical bins")
    # Generate histograms only for columns in both baseline and target dataset
    spark = init_spark()
    schema = StructType(
        [
            StructField("feature_name", StringType(), True),
            StructField("data_type", StringType(), True),
            StructField("bucket", DoubleType(), True),
        ]
    )
    bin_edges = compute_numerical_bins(df1, df2)

    print(f"bin edges: {bin_edges}")
    data = []
    for feature in bin_edges:
        print(bin_edges[feature])
        for i in range(len(bin_edges[feature])):
            print(type(bin_edges[feature][i]))
            data.append(
                {
                    "feature_name": feature,
                    "data_type": "Numerical",
                    "bucket": float(bin_edges[feature][i]),
                }
            )
    return spark.createDataFrame(data=data, schema=schema)

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Entry script for the Compute Histogram component."""

import argparse
from histogram import compute_histograms
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    DoubleType,
)
from shared_utilities.io_utils import init_spark, read_mltable_in_spark, save_spark_df_as_mltable


def _create_empty_histogram_buckets_df():
    spark = init_spark()
    schema = StructType(
        [
            StructField("feature_name", StringType(), True),
            StructField("data_type", StringType(), True),
            StructField("bucket", DoubleType(), True),
        ]
    )
    return spark.createDataFrame(data=[], schema=schema)


def run():
    """Compute histogram."""
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str)
    parser.add_argument("--histogram_buckets", type=str)
    parser.add_argument("--histogram", type=str)
    args = parser.parse_args()

    df = read_mltable_in_spark(args.input_data)
    
    histogram_buckets = None
    try:
        histogram_buckets = read_mltable_in_spark(args.histogram_buckets)
    except Exception:
        print("No histogram buckets detected.")
        histogram_buckets = _create_empty_histogram_buckets_df()

    histogram_df = compute_histograms(df, histogram_buckets)

    save_spark_df_as_mltable(histogram_df, args.histogram)


if __name__ == "__main__":
    run()

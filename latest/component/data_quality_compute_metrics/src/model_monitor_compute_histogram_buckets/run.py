# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Entry script for the Compute Histogram Buckets component."""

import argparse
from histogram_buckets import compute_histogram_buckets
from shared_utilities.io_utils import read_mltable_in_spark, save_spark_df_as_mltable


def run():
    """Compute histogram."""
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data_1", type=str)
    parser.add_argument("--input_data_2", type=str, required=False, nargs="?")
    parser.add_argument("--histogram_buckets", type=str)
    args = parser.parse_args()

    df1 = read_mltable_in_spark(args.input_data_1)
    df2 = df1
    if args.input_data_2 is not None:
        df2 = read_mltable_in_spark(args.input_data_2)

    histogram_buckets = compute_histogram_buckets(df1, df2)
    save_spark_df_as_mltable(histogram_buckets, args.histogram_buckets)


if __name__ == "__main__":
    run()

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Entry script for Data Drift Compute Metrics Spark Component."""

import argparse
from pyspark.sql.types import StringType
from pyspark.sql.functions import regexp_replace
from shared_utilities.io_utils import try_read_mltable_in_spark_with_error, save_spark_df_as_mltable
from compute_data_quality_statistics import compute_data_quality_statistics


def run():
    """Compute metrics."""
    # Parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_data", type=str)
    parser.add_argument("--data_statistics", type=str)
    parser.add_argument("--override_numerical_features", type=str, required=False)
    parser.add_argument("--override_categorical_features", type=str, required=False)
    args = parser.parse_args()

    df = try_read_mltable_in_spark_with_error(args.baseline_data, "baseline_data")

    metric_unique_df = compute_data_quality_statistics(df)

    sp_metric_unique_df = (
        metric_unique_df.to_spark()
    )  # Convert back to Spark dataframe to output as MLTable

    # CONVERT TO STRING
    sp_metric_unique_df = sp_metric_unique_df.withColumn(
        "set", sp_metric_unique_df["set"].cast(StringType())
    )
    # remove brackets as they will get added again with array type conversion
    sp_metric_unique_df = sp_metric_unique_df.withColumn(
        "set", regexp_replace("set", "\\[|\\]", "")
    )

    # Save metrics in default blob store and log it in active run
    save_spark_df_as_mltable(sp_metric_unique_df, args.data_statistics)


if __name__ == "__main__":
    run()

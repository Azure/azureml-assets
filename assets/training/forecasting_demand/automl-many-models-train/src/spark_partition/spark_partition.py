# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Spark partition component code."""
import argparse
import uuid

from pyspark.sql import SparkSession


def parse_spark_partition_args():
    """Parse args for spark partition."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data")
    parser.add_argument("--partitioned_data")
    parser.add_argument("--partition_column_names")
    parser.add_argument("--input_type")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    print("Start spark partition.")
    args = parse_spark_partition_args()
    partition_column_names = args.partition_column_names.split()

    print("Input:")
    print(f"Inputs path {args.raw_data}")
    print(f"Use input type as {args.input_type}")
    print(f"partition data path {args.partitioned_data}")
    print(f"partition column name(s) {partition_column_names}")
    is_csv = args.input_type == "csv"

    spark = SparkSession.builder.appName("SparkPartition").getOrCreate()

    raw_data = args.raw_data
    new_partition_columns = []

    if is_csv:
        print("read csv data")
        sdf = spark.read.option("header", "true").option("inferschema", "true").csv(raw_data)
    else:
        print("read parquet data")
        sdf = spark.read.parquet(raw_data)

    sdf.printSchema()

    for old_col in partition_column_names:
        new_col = str(uuid.uuid4())
        new_partition_columns.append(new_col)
        sdf = sdf.withColumn(new_col,  getattr(sdf, old_col))

    sdf.write.option("header", True).partitionBy(new_partition_columns).mode("overwrite").parquet(args.partitioned_data)
    print("Done")

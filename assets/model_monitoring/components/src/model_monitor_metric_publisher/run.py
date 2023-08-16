# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Entry script for Model Monitor Azure Monitor Metric Publisher Component."""

import argparse

from pyspark.sql import Row
from typing import List

from azure_monitor_metric_publisher import publish_metric
from shared_utilities.io_utils import read_mltable_in_spark


def run():
    """Publisher Azure monitor metrics."""
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--signal_metrics", type=str)
    arg_parser.add_argument("--monitor_name", type=str)
    arg_parser.add_argument("--signal_name", type=str)
    arg_parser.add_argument("--location", type=str)
    arg_parser.add_argument("--workspace_resource_id", type=str)
    args = arg_parser.parse_args()

    metrics: List[Row] = read_mltable_in_spark(args.signal_metrics).collect()

    publish_metric(metrics, args.monitor_name, args.signal_name, args.location, args.workspace_resource_id)

    print("*************** Publish metrics ***************")
    print("Successfully executed the metric publisher component.")

if __name__ == "__main__":
    run()

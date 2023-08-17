# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Entry script for Model Monitor Azure Monitor Metric Publisher Component."""

import argparse
import os
import re

from pyspark.sql import Row
from typing import List

from azure_monitor_metric_publisher import publish_metric
from shared_utilities.datetime_utils import parse_datetime_from_string
from shared_utilities.io_utils import read_mltable_in_spark
from shared_utilities.log_utils import log_error


def run():
    """Publisher Azure monitor metrics."""
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--signal_metrics", type=str)
    arg_parser.add_argument("--monitor_name", type=str)
    arg_parser.add_argument("--signal_name", type=str)
    arg_parser.add_argument("--data_window_start", type=str)
    arg_parser.add_argument("--data_window_end", type=str)
    args = arg_parser.parse_args()

    metrics: List[Row] = read_mltable_in_spark(args.signal_metrics).collect()

    try:
        azureml_service_endpoint = os.environ['AZUREML_SERVICE_ENDPOINT']
        workspace_resource_id = os.environ['AZUREML_WORKSPACE_SCOPE']
    except KeyError as err:
        log_error(f"Failed to find a required environment variable, error: {str(err)}.")
        raise

    location_search = re.search('https://([A-Za-z0-9]+).api.azureml.ms', azureml_service_endpoint, re.IGNORECASE)
    if location_search:
        location = location_search.group(1)
    else:
        message = f"Failed to extract location string. "\
                  + f"Value of environment variable 'AZUREML_SERVICE_ENDPOINT': {azureml_service_endpoint}"
        log_error(message)
        raise ValueError(message)

    format_data = "%Y-%m-%d %H:%M:%S"
    data_window_start = parse_datetime_from_string(format_data, args.data_window_start)
    data_window_end = parse_datetime_from_string(format_data, args.data_window_end)

    publish_metric(
        metrics,
        args.monitor_name,
        args.signal_name,
        data_window_start,
        data_window_end,
        location,
        workspace_resource_id)

    print("*************** Publish metrics ***************")
    print("Successfully executed the metric publisher component.")


if __name__ == "__main__":
    run()

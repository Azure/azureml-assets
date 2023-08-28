# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Entry script for Compute Performance Metrics Component."""

import argparse
import json
from typing import List
from datetime import datetime

import numpy as np
import pandas as pd
import mlflow
from azureml._common._error_definition.azureml_error import AzureMLError

from utils.helper import get_logger
from utils.io import resolve_io_path, read_jsonl_files
from utils.exceptions import (
    swallow_all_exceptions,
    BenchmarkValidationException,
    MissingColumnException,
)
from utils.error_definitions import BenchmarkValidationError, MissingColumnError


logger = get_logger(__name__)


def extract_and_validate_percentiles(percentile_arr: List[str]) -> List[float]:
    """Extract the percentiles as floats, erroring if any of them are invalid."""
    validated_percentiles = []
    for percentile in percentile_arr:
        valid_percentile = True
        try:
            new_percentile = float(percentile)
            if new_percentile >= 0 and new_percentile <= 100:
                validated_percentiles.append(new_percentile)
            else:
                valid_percentile = False
        except ValueError:
            valid_percentile = False

        if not valid_percentile:
            mssg = f"'{percentile}' was provided as a percentile but is not a valid percentile."
            raise BenchmarkValidationException._with_error(
                AzureMLError.create(BenchmarkValidationError, error_details=mssg)
            )

    return validated_percentiles


def validate_column_args(args: argparse.Namespace, data: pd.DataFrame) -> None:
    """Validate that the column names provided exist in the data."""
    if args.batch_size_column_name not in data.columns:
        mssg = (
            f"'{args.batch_size_column_name}' was provided as the batch size column but no such column exists "
            "in the provided data. "
        )
        raise MissingColumnException._with_error(
            AzureMLError.create(MissingColumnError, error_details=mssg)
        )
    if args.start_time_column_name not in data.columns:
        mssg = (
            f"'{args.start_time_column_name}' was provided as the start time column but no such column exists "
            "in the provided data. "
        )
        raise MissingColumnException._with_error(
            AzureMLError.create(MissingColumnError, error_details=mssg)
        )
    if args.end_time_column_name not in data.columns:
        mssg = (
            f"'{args.end_time_column_name}' was provided as the end time column but no such column exists "
            "in the provided data. "
        )
        raise MissingColumnException._with_error(
            AzureMLError.create(MissingColumnError, error_details=mssg)
        )

    if (
        args.input_token_count_column_name is not None
        and args.input_token_count_column_name not in data.columns
    ):
        mssg = (
            f"'{args.input_token_count_column_name}' was provided as the input token column but no such column "
            "exists in the provided data. "
        )
        raise MissingColumnException._with_error(
            AzureMLError.create(MissingColumnError, error_details=mssg)
        )
    if (
        args.output_token_count_column_name is not None
        and args.output_token_count_column_name not in data.columns
    ):
        mssg = (
            f"'{args.output_token_count_column_name}' was provided as the output token column but no such column "
            "exists in the provided data. "
        )
        raise MissingColumnException._with_error(
            AzureMLError.create(MissingColumnError, error_details=mssg)
        )
    if (
        args.input_char_count_column_name is not None
        and args.input_char_count_column_name not in data.columns
    ):
        mssg = (
            f"'{args.input_char_count_column_name}' was provided as the input character column but no such "
            "column exists in the provided data. "
        )
        raise MissingColumnException._with_error(
            AzureMLError.create(MissingColumnError, error_details=mssg)
        )
    if (
        args.output_char_count_column_name is not None
        and args.output_char_count_column_name not in data.columns
    ):
        mssg = (
            f"'{args.output_char_count_column_name}' was provided as the output character column but no such "
            "column exists in the provided data. "
        )
        raise MissingColumnException._with_error(
            AzureMLError.create(MissingColumnError, error_details=mssg)
        )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--performance_data", type=str, help="path to model inference performance data"
    )
    parser.add_argument(
        "--percentiles",
        type=lambda x: [i.strip() for i in x.split(",") if i and not i.isspace()],
        dest="percentiles",
        required=False,
        default=[],
    )
    parser.add_argument("--batch_size_column_name", type=str)
    parser.add_argument("--start_time_column_name", type=str)
    parser.add_argument("--end_time_column_name", type=str)
    parser.add_argument(
        "--input_token_count_column_name", type=str, required=False, default=None
    )
    parser.add_argument(
        "--output_token_count_column_name", type=str, required=False, default=None
    )
    parser.add_argument(
        "--input_char_count_column_name", type=str, required=False, default=None
    )
    parser.add_argument(
        "--output_char_count_column_name", type=str, required=False, default=None
    )
    parser.add_argument(
        "--performance_result",
        type=str,
        help="path to store performance metric results",
    )

    args, _ = parser.parse_known_args()
    logger.info(f"Arguments: {args}")
    return args


@swallow_all_exceptions(logger)
def main(args: argparse.Namespace) -> None:
    """
    Entry function for Compute Performance Metrics Component.

    :param args: Command-line arguments
    :return: None
    """
    # Validate args
    percentiles = extract_and_validate_percentiles(args.percentiles)

    # Load data and then validate column names
    input_file_paths = resolve_io_path(args.performance_data)
    all_data = pd.DataFrame(read_jsonl_files(input_file_paths))
    validate_column_args(args, all_data)

    # Start Logging
    mlflow.start_run()
    results = dict()

    if len(all_data) > 0:
        # Calculate latency data
        start_data = np.array(
            [
                datetime.fromisoformat(t).timestamp()
                for t in list(all_data[args.start_time_column_name])
            ]
        )
        end_data = np.array(
            [
                datetime.fromisoformat(t).timestamp()
                for t in list(all_data[args.end_time_column_name])
            ]
        )
        latency_data = (end_data - start_data) * 1000
        latency_data = latency_data / np.array(all_data[args.batch_size_column_name])
        results["latency_avg"] = np.average(latency_data)

        client = mlflow.tracking.MlflowClient()
        for percentile in percentiles:
            results["latency_p{0}".format(str(percentile))] = np.percentile(
                latency_data, percentile
            )

        # Get input character data if it exists
        if args.input_char_count_column_name is not None:
            input_char_data = np.array(all_data[args.input_char_count_column_name])
            input_char_normalized_latency = latency_data / input_char_data
            results["latency_per_input_char_avg"] = np.average(
                input_char_normalized_latency
            )

            for percentile in percentiles:
                results[
                    "latency_per_input_char_p{0}".format(str(percentile))
                ] = np.percentile(input_char_normalized_latency, percentile)

            # Log latency versus characters
            client.log_batch(
                mlflow.active_run().info.run_id,
                metrics=[
                    mlflow.entities.Metric(
                        key="latency_vs_input_char",
                        value=val[0],
                        timestamp=0,
                        step=int(val[1]),
                    )
                    for val in zip(latency_data, input_char_data)
                ],
            )

        # Get output character data if it exists
        if args.output_char_count_column_name is not None:
            output_char_data = np.array(all_data[args.output_char_count_column_name])
            output_char_normalized_latency = latency_data / output_char_data
            results["latency_per_output_char_avg"] = np.average(
                output_char_normalized_latency
            )

            for percentile in percentiles:
                results[
                    "latency_per_output_char_p{0}".format(str(percentile))
                ] = np.percentile(output_char_normalized_latency, percentile)

            # Log latency versus characters
            client.log_batch(
                mlflow.active_run().info.run_id,
                metrics=[
                    mlflow.entities.Metric(
                        key="latency_vs_output_char",
                        value=val[0],
                        timestamp=0,
                        step=int(val[1]),
                    )
                    for val in zip(latency_data, output_char_data)
                ],
            )

        # Get input plus output character data if they both exists
        if (
            args.input_char_count_column_name is not None
            and args.output_char_count_column_name is not None
        ):
            input_char_data = np.array(all_data[args.input_char_count_column_name])
            output_char_data = np.array(all_data[args.output_char_count_column_name])
            input_output_char_data = input_char_data + output_char_data
            input_output_char_normalized_latency = latency_data / input_output_char_data
            results["latency_per_input_output_char_avg"] = np.average(
                input_output_char_normalized_latency
            )

            for percentile in percentiles:
                results[
                    "latency_per_input_output_char_p{0}".format(str(percentile))
                ] = np.percentile(input_output_char_normalized_latency, percentile)

            # Log latency versus characters
            client.log_batch(
                mlflow.active_run().info.run_id,
                metrics=[
                    mlflow.entities.Metric(
                        key="latency_vs_input_output_char",
                        value=val[0],
                        timestamp=0,
                        step=int(val[1]),
                    )
                    for val in zip(latency_data, input_output_char_data)
                ],
            )

        # Get input token data if it exists
        if args.input_token_count_column_name is not None:
            input_token_data = np.array(all_data[args.input_token_count_column_name])

            input_token_count = np.sum(input_token_data)
            results["total_input_tokens"] = int(input_token_count)
            total_latency = np.sum(latency_data)
            results["input_token_throughput"] = input_token_count / total_latency

            input_token_normalized_latency = latency_data / input_token_data
            results["latency_per_input_token_avg"] = np.average(
                input_token_normalized_latency
            )

            for percentile in percentiles:
                results[
                    "latency_per_input_token_p{0}".format(str(percentile))
                ] = np.percentile(input_token_normalized_latency, percentile)

            # Log latency versus tokens
            client.log_batch(
                mlflow.active_run().info.run_id,
                metrics=[
                    mlflow.entities.Metric(
                        key="latency_vs_input_tokens",
                        value=val[0],
                        timestamp=0,
                        step=int(val[1]),
                    )
                    for val in zip(latency_data, input_token_data)
                ],
            )

        # Get output token data if it exists
        if args.output_token_count_column_name is not None:
            output_token_data = np.array(all_data[args.output_token_count_column_name])

            output_token_count = np.sum(output_token_data)
            results["total_output_tokens"] = int(output_token_count)
            total_latency = np.sum(latency_data)
            results["output_token_throughput"] = output_token_count / total_latency

            output_token_normalized_latency = latency_data / output_token_data
            results["latency_per_output_token_avg"] = np.average(
                output_token_normalized_latency
            )

            for percentile in percentiles:
                results[
                    "latency_per_output_token_p{0}".format(str(percentile))
                ] = np.percentile(output_token_normalized_latency, percentile)

            # Log latency versus tokens
            client.log_batch(
                mlflow.active_run().info.run_id,
                metrics=[
                    mlflow.entities.Metric(
                        key="latency_vs_output_tokens",
                        value=val[0],
                        timestamp=0,
                        step=int(val[1]),
                    )
                    for val in zip(latency_data, output_token_data)
                ],
            )

        # Get input plus output token data if they both exists
        if (
            args.input_token_count_column_name is not None
            and args.output_token_count_column_name is not None
        ):
            input_token_data = np.array(all_data[args.input_token_count_column_name])
            output_token_data = np.array(all_data[args.output_token_count_column_name])
            input_output_token_data = input_token_data + output_token_data
            input_output_token_normalized_latency = (
                latency_data / input_output_token_data
            )
            results["latency_per_input_output_token_avg"] = np.average(
                input_output_token_normalized_latency
            )

            for percentile in percentiles:
                results[
                    "latency_per_input_output_token_p{0}".format(str(percentile))
                ] = np.percentile(input_output_token_normalized_latency, percentile)

            # Log latency versus tokens
            client.log_batch(
                mlflow.active_run().info.run_id,
                metrics=[
                    mlflow.entities.Metric(
                        key="latency_vs_input_output_tokens",
                        value=val[0],
                        timestamp=0,
                        step=int(val[1]),
                    )
                    for val in zip(latency_data, input_output_token_data)
                ],
            )

    # Output the metrics that are logged in the metrics file
    mlflow.log_metrics(results)

    # Save the metrics
    with open(args.performance_result, "w") as f:
        json.dump(results, f)

    # Stop Logging
    mlflow.end_run()


if __name__ == "__main__":
    args = parse_args()
    main(args)

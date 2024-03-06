# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Entry script for Compute Performance Metrics Component."""

import argparse
import json
from typing import List, Optional
from datetime import datetime

import numpy as np
import pandas as pd
import mlflow
from azureml._common._error_definition.azureml_error import AzureMLError

from aml_benchmark.utils.logging import get_logger
from aml_benchmark.utils.io import resolve_io_path, read_jsonl_files
from aml_benchmark.utils.exceptions import (
    swallow_all_exceptions,
    BenchmarkValidationException,
    MissingColumnException,
)
from aml_benchmark.utils.aml_run_utils import str2bool
from aml_benchmark.utils.error_definitions import BenchmarkValidationError, MissingColumnError


logger = get_logger(__name__)
MAX_ALLOWED_LENGTH = 50


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


def validate_column_args(
    data: pd.DataFrame,
    batch_size_column_name: str,
    start_time_column_name: str,
    end_time_column_name: str,
    input_token_count_column_name: Optional[str] = None,
    output_token_count_column_name: Optional[str] = None,
    input_char_count_column_name: Optional[str] = None,
    output_char_count_column_name: Optional[str] = None,
) -> None:
    """
    Validate that the column names provided exist in the data.

    :param data: Dataframe containing the performance data.
    :param batch_size_column_name: Name of the column in the performance data that contains the batch size info.
    :param start_time_column_name: Name of the column in the performance data that contains the start timestamp in
    ISO 8601 format.
    :param end_time_column_name: Name of the column in the performance data that contains the end timestamp in
    ISO 8601 format.
    :param input_token_count_column_name: Name of the column in the performance data that contains the
    input token count info.
    :param output_token_count_column_name: Name of the column in the performance data that contains the
    output token count info.
    :param input_char_count_column_name: Name of the column in the performance data that contains the
    input character count info.
    :param output_char_count_column_name: Name of the column in the performance data that contains the
    output character count info.
    :return: None
    """
    if batch_size_column_name not in data.columns:
        mssg = (
            f"'{batch_size_column_name}' was provided as the batch size column but no such column exists "
            "in the provided data. "
        )
        raise MissingColumnException._with_error(
            AzureMLError.create(MissingColumnError, error_details=mssg)
        )
    if start_time_column_name not in data.columns:
        mssg = (
            f"'{start_time_column_name}' was provided as the start time column but no such column exists "
            "in the provided data. "
        )
        raise MissingColumnException._with_error(
            AzureMLError.create(MissingColumnError, error_details=mssg)
        )
    if end_time_column_name not in data.columns:
        mssg = (
            f"'{end_time_column_name}' was provided as the end time column but no such column exists "
            "in the provided data. "
        )
        raise MissingColumnException._with_error(
            AzureMLError.create(MissingColumnError, error_details=mssg)
        )

    if (
        input_token_count_column_name is not None
        and input_token_count_column_name not in data.columns
    ):
        mssg = (
            f"'{input_token_count_column_name}' was provided as the input token column but no such column "
            "exists in the provided data. "
        )
        raise MissingColumnException._with_error(
            AzureMLError.create(MissingColumnError, error_details=mssg)
        )
    if (
        output_token_count_column_name is not None
        and output_token_count_column_name not in data.columns
    ):
        mssg = (
            f"'{output_token_count_column_name}' was provided as the output token column but no such column "
            "exists in the provided data. "
        )
        raise MissingColumnException._with_error(
            AzureMLError.create(MissingColumnError, error_details=mssg)
        )
    if (
        input_char_count_column_name is not None
        and input_char_count_column_name not in data.columns
    ):
        mssg = (
            f"'{input_char_count_column_name}' was provided as the input character column but no such "
            "column exists in the provided data. "
        )
        raise MissingColumnException._with_error(
            AzureMLError.create(MissingColumnError, error_details=mssg)
        )
    if (
        output_char_count_column_name is not None
        and output_char_count_column_name not in data.columns
    ):
        mssg = (
            f"'{output_char_count_column_name}' was provided as the output character column but no such "
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
        "--is_batch_inference_result", default=True, type=str2bool,
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
def main(
    performance_result: str,
    performance_data: str,
    percentiles: List[float],
    batch_size_column_name: str,
    start_time_column_name: str,
    end_time_column_name: str,
    input_token_count_column_name: Optional[str] = None,
    output_token_count_column_name: Optional[str] = None,
    input_char_count_column_name: Optional[str] = None,
    output_char_count_column_name: Optional[str] = None,
    is_batch_inference_result: bool = True
) -> None:
    """
    Entry function for Compute Performance Metrics Component.

    :param performance_result: Path to the file where the calculated performance metric results will be saved.
    :param performance_data: Path to the data outputted by model inferencing that contains performance data.
    :param percentiles: List of percentiles of latency to be calculated.
    :param batch_size_column_name: Name of the column in the performance data that contains the batch size info.
    :param start_time_column_name: Name of the column in the performance data that contains the start timestamp in
    ISO 8601 format.
    :param end_time_column_name: Name of the column in the performance data that contains the end timestamp in
    ISO 8601 format.
    :param input_token_count_column_name: Name of the column in the performance data that contains the
    input token count info.
    :param output_token_count_column_name: Name of the column in the performance data that contains the
    output token count info.
    :param input_char_count_column_name: Name of the column in the performance data that contains the
    input character count info.
    :param is_batch_inference_result: Whether the performance data is from batch inference.
    :param output_char_count_column_name: Name of the column in the performance data that contains the
    output character count info.
    :return: None
    """
    # Validate args
    percentiles = extract_and_validate_percentiles(percentiles)

    # Load data and then validate column names
    input_file_paths = resolve_io_path(performance_data)
    all_data = pd.DataFrame(read_jsonl_files(input_file_paths))
    validate_column_args(
        data=all_data,
        batch_size_column_name=batch_size_column_name,
        start_time_column_name=start_time_column_name,
        end_time_column_name=end_time_column_name,
        input_token_count_column_name=input_token_count_column_name,
        output_token_count_column_name=output_token_count_column_name,
        input_char_count_column_name=input_char_count_column_name,
        output_char_count_column_name=output_char_count_column_name,
    )

    # Start Logging
    mlflow.start_run()
    results = dict()
    run_timespan = None

    if len(all_data) > 0:
        # Calculate latency data
        start_data = np.array(
            [
                datetime.fromisoformat(t).timestamp()
                for t in list(all_data[start_time_column_name])
            ]
        )
        end_data = np.array(
            [
                datetime.fromisoformat(t).timestamp()
                for t in list(all_data[end_time_column_name])
            ]
        )
        latency_data = (end_data - start_data) * 1000
        run_timespan = np.max(end_data) - np.min(start_data)
        latency_data = latency_data / np.array(all_data[batch_size_column_name])
        results["latency_avg"] = np.average(latency_data)
        total_latency = np.sum(latency_data)
        if is_batch_inference_result:
            results["requests_per_sec"] = len(all_data) / run_timespan
        else:
            results["requests_per_sec"] = len(all_data) / total_latency

        client = mlflow.tracking.MlflowClient()
        for percentile in percentiles:
            results["latency_p{0}".format(str(percentile))] = np.percentile(
                latency_data, percentile
            )

        # Get input character data if it exists
        if input_char_count_column_name is not None:
            input_char_data = np.array(all_data[input_char_count_column_name])
            input_char_normalized_latency = latency_data / input_char_data
            results["latency_per_input_char_avg"] = np.average(
                input_char_normalized_latency
            )

            for percentile in percentiles:
                results[
                    "latency_per_input_char_p{0}".format(str(percentile))
                ] = np.percentile(input_char_normalized_latency, percentile)

            if len(latency_data) <= MAX_ALLOWED_LENGTH:
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
        if output_char_count_column_name is not None:
            output_char_data = np.array(all_data[output_char_count_column_name])
            output_char_normalized_latency = latency_data / output_char_data
            results["latency_per_output_char_avg"] = np.average(
                output_char_normalized_latency
            )

            for percentile in percentiles:
                results[
                    "latency_per_output_char_p{0}".format(str(percentile))
                ] = np.percentile(output_char_normalized_latency, percentile)

            if len(latency_data) <= MAX_ALLOWED_LENGTH:
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
            input_char_count_column_name is not None
            and output_char_count_column_name is not None
        ):
            input_char_data = np.array(all_data[input_char_count_column_name])
            output_char_data = np.array(all_data[output_char_count_column_name])
            input_output_char_data = input_char_data + output_char_data
            input_output_char_normalized_latency = latency_data / input_output_char_data
            results["latency_per_input_output_char_avg"] = np.average(
                input_output_char_normalized_latency
            )

            for percentile in percentiles:
                results[
                    "latency_per_input_output_char_p{0}".format(str(percentile))
                ] = np.percentile(input_output_char_normalized_latency, percentile)

            if len(latency_data) <= MAX_ALLOWED_LENGTH:
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
        if input_token_count_column_name is not None:
            input_token_data = np.array(all_data[input_token_count_column_name])

            input_token_count = np.sum(input_token_data)
            results["total_input_tokens"] = int(input_token_count)
            total_latency = np.sum(latency_data)
            if is_batch_inference_result:
                results["input_tokens_per_sec"] = input_token_count / run_timespan
            else:
                results["input_tokens_per_sec"] = input_token_count / total_latency
            input_token_normalized_latency = latency_data / input_token_data

            results["latency_per_input_token_avg"] = np.average(
                input_token_normalized_latency
            )

            for percentile in percentiles:
                results[
                    "latency_per_input_token_p{0}".format(str(percentile))
                ] = np.percentile(input_token_normalized_latency, percentile)

            if len(latency_data) <= MAX_ALLOWED_LENGTH:
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
        if output_token_count_column_name is not None:
            output_token_data = np.array(all_data[output_token_count_column_name])

            output_token_count = np.sum(output_token_data)
            results["total_output_tokens"] = int(output_token_count)
            total_latency = np.sum(latency_data)
            if is_batch_inference_result:
                results["output_tokens_per_sec"] = output_token_count / run_timespan
            else:
                results["output_tokens_per_sec"] = output_token_count / total_latency

            output_token_normalized_latency = latency_data / output_token_data
            results["latency_per_output_token_avg"] = np.average(
                output_token_normalized_latency
            )

            for percentile in percentiles:
                results[
                    "latency_per_output_token_p{0}".format(str(percentile))
                ] = np.percentile(output_token_normalized_latency, percentile)

            if len(latency_data) <= MAX_ALLOWED_LENGTH:
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
            input_token_count_column_name is not None
            and output_token_count_column_name is not None
        ):
            input_token_data = np.array(all_data[input_token_count_column_name])
            output_token_data = np.array(all_data[output_token_count_column_name])
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

            if len(latency_data) <= MAX_ALLOWED_LENGTH:
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
    with open(performance_result, "w") as f:
        json.dump(results, f)

    # Stop Logging
    mlflow.end_run()


if __name__ == "__main__":
    args = parse_args()
    main(
        performance_result=args.performance_result,
        performance_data=args.performance_data,
        percentiles=args.percentiles,
        batch_size_column_name=args.batch_size_column_name,
        start_time_column_name=args.start_time_column_name,
        end_time_column_name=args.end_time_column_name,
        input_token_count_column_name=args.input_token_count_column_name,
        output_token_count_column_name=args.output_token_count_column_name,
        input_char_count_column_name=args.input_char_count_column_name,
        output_char_count_column_name=args.output_char_count_column_name,
        is_batch_inference_result=args.is_batch_inference_result
    )

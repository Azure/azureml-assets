# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Entry script for Endpoint Inference Triton Component."""

from typing import List, Optional, Dict, Any
import argparse
import json

from azureml._common._error_definition.azureml_error import AzureMLError

from aml_benchmark.utils.io import resolve_io_path, read_jsonl_files, parse_jinja_template, write_jsonl_file
from aml_benchmark.utils.logging import get_logger, log_mlflow_params
from aml_benchmark.utils.exceptions import swallow_all_exceptions, BenchmarkValidationException
from aml_benchmark.utils.error_definitions import BenchmarkValidationError
from aml_benchmark.utils.constants import TaskType
from .payload_preparer import prepare_payload
from .infer import get_predictions


logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=f"{__file__}")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to the input .jsonl file(s).",
    )
    parser.add_argument(
        "--task_type",
        type=str,
        required=True,
        help="Task type, either `text_generation` or `chat_completion`.",
        choices=[member.value for member in TaskType],
    )
    parser.add_argument(
        "--input_pattern",
        type=str,
        required=True,
        help="Jinja template that is used to generate payload.",
    )
    parser.add_argument(
        "--endpoint_url",
        type=str,
        required=True,
        help="The endpoint url.",
    )
    parser.add_argument(
        "--deployment_name",
        type=str,
        required=True,
        help="The deployment name.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="The model name.",
    )
    parser.add_argument(
        "--model_version",
        type=str,
        required=True,
        help="The model version.",
        default="1",
    )
    parser.add_argument(
        "--connections_name",
        type=str,
        required=True,
        help="The connections name that has the API Key to successfully authenticate the endpoint.",
    )
    parser.add_argument(
        "--label_column_name",
        type=str,
        required=True,
        help="The label column name.",
    )
    parser.add_argument(
        "--predictions", type=str, required=True, help="Path to the predictions file."
    )
    parser.add_argument(
        "--performance_metadata", type=str, required=True, help="Path to the performance metadata file."
    )
    parser.add_argument(
        "--ground_truth", type=str, required=True, help="Path to the ground truth file."
    )

    args, _ = parser.parse_known_args()
    logger.info(f"Arguments: {args}")
    return args


@swallow_all_exceptions(logger)
def main(
    dataset: str,
    input_pattern: str,
    task_type: str,
    endpoint_url: str,
    deployment_name: str,
    model_name: str,
    connections_name: str,
    label_column_name: str,
    predictions: str,
    ground_truth: str,
    performance_metadata: str,
    model_version: str = "1",
) -> None:
    """
    Entry function for Endpoint Inference Triton Component.

    :param dataset: The input dataset which can be `MLTable`, `uri_folder`, or `uri_file`.
    :param task_type: The task type, either `text_generation` or `chat_completion`.
    :param input_pattern: The jinja template that is used to generate payload.
    :param endpoint_url: The endpoint url.
    :param deployment_name: The deployment name.
    :param model_name: The model name.
    :param connections_name: The connections name that has the API Key to successfully authneticate the endpoint.
    :param label_column_name: The label column name.
    :param predictions: Path to the predictions jsonl file.
    :param ground_truth: Path to the ground truth jsonl file.
    :param performance_metadata: Path to the performance metadata jsonl file.
    :param model_version: The model version, defaults to "1".
    :return: None
    """
    input_file_paths = [file for file in resolve_io_path(dataset) if file.endswith(".jsonl")]
    if not input_file_paths:
        raise BenchmarkValidationException._with_error(
            AzureMLError.create(BenchmarkValidationError, error_details="No .jsonl files found in input dataset.")
        )
    input_data: List[Dict[str, Any]] = read_jsonl_files(input_file_paths)

    parsed_input_data = [json.loads(parse_jinja_template(input_pattern, data)) for data in input_data]
    payload = prepare_payload(parsed_input_data, task_type)
    pred_data, perf_data = get_predictions(
        task_type=task_type,
        endpoint_url=endpoint_url,
        deployment_name=deployment_name,
        model_name=model_name,
        model_version=model_version,
        connections_name=connections_name,
        payload=payload,
    )

    write_jsonl_file(pred_data, predictions)
    
    ground_truth_data = [{label_column_name: data[label_column_name]} for data in input_data]
    write_jsonl_file(ground_truth_data, ground_truth)

    write_jsonl_file(perf_data, performance_metadata)

    log_mlflow_params(
        dataset=input_file_paths,
        input_pattern=input_pattern,
        label_column_name=label_column_name
    )


if __name__ == "__main__":
    args = parse_args()
    main(
        dataset=args.dataset,
        task_type=args.task_type,
        input_pattern=args.input_pattern,
        endpoint_url=args.endpoint_url,
        deployment_name=args.deployment_name,
        model_name=args.model_name,
        model_version=args.model_version,
        connections_name=args.connections_name,
        label_column_name=args.label_column_name,
        predictions=args.predictions,
        performance_metadata=args.performance_metadata,
        ground_truth=args.ground_truth,
    )

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Entry script for batch output formatter component."""

import argparse
import os
from typing import Optional, List, Dict, Any
import pandas as pd

from aml_benchmark.utils.io import resolve_io_path, read_jsonl_files, save_list_to_jsonl_if_path_provided
from aml_benchmark.utils.logging import get_logger, log_params_and_metrics
from aml_benchmark.utils.exceptions import swallow_all_exceptions
from aml_benchmark.utils.aml_run_utils import str2bool
from aml_benchmark.utils.online_endpoint.online_endpoint_model import OnlineEndpointModel
from aml_benchmark.utils.online_endpoint.endpoint_utils import EndpointUtilities
from aml_benchmark.utils.exceptions import BenchmarkUserException
from aml_benchmark.utils.error_definitions import BenchmarkUserError
from azureml._common._error_definition.azureml_error import AzureMLError
from .result_converters import ResultConverters


logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse the args for the method."""
    # Input and output arguments
    parser = argparse.ArgumentParser(description=f"{__file__}")
    parser.add_argument("--model_type", type=str, help="model_type", default=None)
    parser.add_argument("--batch_inference_output", type=str, help="path to prompt crafter output")
    parser.add_argument("--prediction_data", type=str, help="path to output location")
    parser.add_argument("--successful_requests_data", type=str, required=False, help="path to failed requests output")
    parser.add_argument("--failed_requests_data", type=str, required=False, help="path to failed requests output")
    parser.add_argument("--blocked_requests_data", type=str, required=False, help="path to blocked requests output")
    parser.add_argument("--ground_truth_input", type=str, help="path to output location", default=None)
    parser.add_argument(
        "--predict_ground_truth_data", type=str,
        help="The ground truth data mapping 1-1 to the prediction data.")

    parser.add_argument("--perf_data", type=str, help="path to output location")
    parser.add_argument("--endpoint_url", type=str, help="endpoint_url", default=None)
    parser.add_argument("--metadata_key", type=str, help="metadata key", default=None)
    parser.add_argument("--data_id_key", type=str, help="metadata key", default=None)
    parser.add_argument("--label_key", type=str, help="label key")
    parser.add_argument("--additional_columns", type=str, help="additional columns")
    parser.add_argument("--handle_response_failure", type=str, help="how to handler failed response.")
    parser.add_argument("--fallback_value", type=str, help="The fallback value.", default='')
    parser.add_argument("--is_performance_test", default=False, type=str2bool, help="is_performance_test")
    parser.add_argument(
        "--use_tiktoken",
        type=str2bool,
        default=False,
        help=("If true, `cl100k_base` encoder is used from tiktoken to calculate token count; "
              "overrides any other token count calculation."))
    parser.add_argument("--min_endpoint_success_ratio", default=0, type=float, help="Min success ratio.")
    args, _ = parser.parse_known_args()
    logger.info(f"Arguments: {args}")
    return args


def _reorder_batch_score_result(
    batch_score_result: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]],
    model_type: str,
) -> List[Dict[str, Any]]:
    """Reorder the batch score result based on the ground truth."""
    model = OnlineEndpointModel(model=None, model_version=None, model_type=model_type)
    batch_score_dict = {
        EndpointUtilities.hash_payload_prompt(row["request"], model): row for row in batch_score_result
    }

    batch_score_result = []
    for ground_truth_dict in ground_truth:
        hash_val = ground_truth_dict["payload_id"]
        batch_score_result.append(batch_score_dict[hash_val])

    return batch_score_result


@swallow_all_exceptions(logger)
def main(
        batch_inference_output: str,
        model_type: str,
        data_id_key: str,
        metadata_key: str,
        label_key: str,
        additional_columns: str,
        ground_truth_input: str,
        prediction_path: str,
        perf_path: str,
        ground_truth_path: str,
        handle_response_failure: str,
        fallback_value: str,
        is_performance_test: bool,
        endpoint_url: str,
        use_tiktoken: bool = False,
        successful_requests_path: Optional[str] = None,
        failed_requests_path: Optional[str] = None,
        blocked_requests_path: Optional[str] = None,
        min_endpoint_success_ratio: float = 0.0,
) -> None:
    """
    Entry script for the script.

    :param batch_inference_output: Path to the batch inference output.
    :param data_id_key: If ground_truth_input is provided, data_id_key should be a unique key
        that in the ground_truth_input to identify corresponding the request payload.
    :param metadata_key: The key that contains ground truth in the request payload. If this is
        empty, the `batch_metadata` will be used.
    :param label_key: The key contains ground truth either in the metadata or in the ground_truth_input.
    :param additional columns: name(s) of the column(s) which could be useful for computing certain metrics,
        separated by comma (",").
    :param ground_truth_input: The ground_truth_input which should contains data_id_key and label_key.
    :param prediction_data: The path to the prediction data.
    :param perf_data: The path to the perf data.
    :param predict_ground_truth_data: The ground truth data that correspond to the prediction_data.
    :param handle_response_failure: How to handle the response failure.
    :param fallback_value: The fallback value.
    :param is_performance_test: Whether it is a performance test.
    :param endpoint_url: The endpoint url.
    :param use_tiktoken: If true, `cl100k_base` encoder is used from tiktoken to calculate token count;
    overrides any other token count calculation.
    :param successful_requests_data: The path to the successful requests data.
    :param failed_requests_data: The path to the failed requests data.
    :param blocked_requests_data: The path to the failed requests data.
    :param min_endpoint_success_ratio: Min endpoint success ratio.
    :return: None
    """
    logger.info("Read batch output data now.")
    data_files = [os.path.join(
        batch_inference_output, f
    ) for f in os.listdir(batch_inference_output) if f.endswith("json") or f.endswith("jsonl")
    ]
    logger.info(f"Receiving {len(data_files)} files.")
    batch_score_result: List[Dict[str, Any]] = read_jsonl_files(data_files)

    prediction_list = []
    perf_list = []
    ground_truth_list = []
    failed_response_list = []
    blocked_response_list = []
    successful_response_list = []
    if ground_truth_input:
        input_file_paths = resolve_io_path(ground_truth_input)
        _ground_truth_data = read_jsonl_files(input_file_paths)
        try:
            batch_score_result = _reorder_batch_score_result(batch_score_result, _ground_truth_data, model_type)
            logger.info("Reordered batch score result successfully.")
        except Exception as e:
            logger.warning(
                "Failed to reorder batch score result, falling back to original order. "
                f"This excpetion does not lead to run failure. Exception details:\n{e}"
            )
        ground_truth_df = pd.DataFrame(_ground_truth_data)
        _ground_truth_data = []
    else:
        ground_truth_df = None
    online_model = OnlineEndpointModel(None, None, model_type, endpoint_url=endpoint_url)

    rc = ResultConverters(
        online_model._model_type, metadata_key, data_id_key,
        label_key, additional_columns, ground_truth_df,
        fallback_value=fallback_value, is_performance_test=is_performance_test)
    logger.info("Convert the data now.")

    failed_requests = 0
    successful_requests = 0
    safety_blocked_requests = 0
    for index, row in enumerate(batch_score_result):
        if rc.is_result_content_safety_failure(row):
            # check for safety failure before result success.
            # blocked requests can be fail or success responses.
            safety_blocked_requests += 1
            logger.warning("Met request blocked due to safety at index {}".format(index))
            blocked_response_list.append(row)
            if handle_response_failure == 'neglect':
                continue
        elif not rc.is_result_success(row):
            failed_requests += 1
            logger.warning("Met failed response at index {}".format(index))
            failed_response_list.append(row)
            if handle_response_failure == 'neglect':
                continue
        else:
            successful_requests += 1
            successful_response_list.append(row)
        prediction_list.append(rc.convert_result(row))
        if rc.is_result_success(row):
            # Don't calculate perf for failed requests.
            perf_list.append(rc.convert_result_perf(row, use_tiktoken))
        if not is_performance_test:
            ground_truth_list.append(rc.convert_result_ground_truth(row))
        else:
            logger.info("is performance test")
            ground_truth_list.append({"ground_truth": ''})
    logger.info("Output data now.")

    save_list_to_jsonl_if_path_provided(prediction_list, prediction_path)
    save_list_to_jsonl_if_path_provided(perf_list, perf_path)
    save_list_to_jsonl_if_path_provided(ground_truth_list, ground_truth_path)
    save_list_to_jsonl_if_path_provided(successful_response_list, successful_requests_path)
    save_list_to_jsonl_if_path_provided(failed_response_list, failed_requests_path)
    save_list_to_jsonl_if_path_provided(blocked_response_list, blocked_requests_path)

    total_requests = failed_requests + successful_requests + safety_blocked_requests
    endpoint_success_ratio = (successful_requests + safety_blocked_requests) / total_requests
    logger.info(f"Total requests: {total_requests}")
    logger.info(f"Successful requests: {successful_requests}")
    logger.info(f"Failed requests: {failed_requests}")
    logger.info(f"Unsafe content blocked requests: {safety_blocked_requests}")
    logger.info(
        "Endpoint success ratio "
        f"(successful_requests + safety_blocked_requests) / total_requests): {endpoint_success_ratio}"
    )
    log_params_and_metrics(
        parameters={},
        metrics={
            'failed_requests': failed_requests,
            'successful_requests': successful_requests,
            'total_requests': total_requests,
            'unsafe_content_blocked_requests': safety_blocked_requests,
            'endpoint_success_ratio': endpoint_success_ratio,
        },
        log_to_parent=True,
    )
    if endpoint_success_ratio + 1e-9 < min_endpoint_success_ratio:
        raise BenchmarkUserException._with_error(
            AzureMLError.create(
                BenchmarkUserError,
                error_details=f"""
                Marking run as failed because endpoint success ratio is {endpoint_success_ratio}
                 which is less than {min_endpoint_success_ratio}. Please check why endpoint
                 requests are failing or set a lower value for min_endpoint_success_ratio parameter.
                 Check the logs for failed entries. Check the metrics for failure statistics.
                """
            )
        )


if __name__ == "__main__":
    args = parse_args()
    main(
        batch_inference_output=args.batch_inference_output,
        model_type=args.model_type,
        data_id_key=args.data_id_key,
        metadata_key=args.metadata_key,
        label_key=args.label_key,
        additional_columns=args.additional_columns,
        ground_truth_input=args.ground_truth_input,
        prediction_path=args.prediction_data,
        perf_path=args.perf_data,
        successful_requests_path=args.successful_requests_data,
        failed_requests_path=args.failed_requests_data,
        blocked_requests_path=args.blocked_requests_data,
        ground_truth_path=args.predict_ground_truth_data,
        handle_response_failure=args.handle_response_failure,
        fallback_value=args.fallback_value,
        is_performance_test=args.is_performance_test,
        endpoint_url=args.endpoint_url,
        use_tiktoken=args.use_tiktoken,
        min_endpoint_success_ratio=args.min_endpoint_success_ratio,
    )

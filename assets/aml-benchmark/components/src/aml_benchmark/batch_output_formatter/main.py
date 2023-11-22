# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Entry script for batch output formatter component."""

import argparse
import os
import pandas as pd

from aml_benchmark.utils.io import resolve_io_path, read_jsonl_files
from aml_benchmark.utils.logging import get_logger
from aml_benchmark.utils.exceptions import swallow_all_exceptions
from aml_benchmark.utils.aml_run_utils import str2bool
from aml_benchmark.utils.online_endpoint.endpoint_utils import EndpointUtilities
from aml_benchmark.utils.online_endpoint.online_endpoint_model import OnlineEndpointModel
from .result_converters import ResultConverters


logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse the args for the method."""
    # Input and output arguments
    parser = argparse.ArgumentParser(description=f"{__file__}")
    parser.add_argument("--model_type", type=str, help="model_type", default=None)
    parser.add_argument("--batch_inference_output", type=str, help="path to prompt crafter output")
    parser.add_argument("--prediction_data", type=str, help="path to output location")
    parser.add_argument("--ground_truth_input", type=str, help="path to output location", default=None)
    parser.add_argument(
        "--predict_ground_truth_data", type=str,
        help="The ground truth data mapping 1-1 to the prediction data.")

    parser.add_argument("--perf_data", type=str, help="path to output location")
    parser.add_argument("--endpoint_url", type=str, help="endpoint_url", default=None)
    parser.add_argument("--metadata_key", type=str, help="metadata key", default=None)
    parser.add_argument("--data_id_key", type=str, help="metadata key", default=None)
    parser.add_argument("--label_key", type=str, help="label key")
    parser.add_argument("--handle_response_failure", type=str, help="how to handler failed response.")
    parser.add_argument("--fallback_value", type=str, help="The fallback value.", default='')
    parser.add_argument("--deployment_config_dir", type=str, help="model_type", default=None)
    parser.add_argument("--is_performance_test", default=False, type=str2bool, help="is_performance_test")
    args, _ = parser.parse_known_args()
    logger.info(f"Arguments: {args}")
    return args


@swallow_all_exceptions(logger)
def main(
        batch_inference_output: str,
        model_type: str,
        data_id_key: str,
        metadata_key: str,
        label_key: str,
        ground_truth_input: str,
        prediction_data: str,
        perf_data: str,
        predict_ground_truth_data: str,
        handle_response_failure: str,
        fallback_value: str,
        is_performance_test: bool,
        endpoint_url: str,
        deployment_config_dir: str
) -> None:
    """
    Entry script for the script.

    :param batch_inference_output: Path to the batch inference output.
    :param data_id_key: If ground_truth_input is provided, data_id_key should be a unique key
        that in the ground_truth_input to identify corresponding the request payload.
    :param metadata_key: The key that contains ground truth in the request payload. If this is
        empty, the `batch_metadata` will be used.
    :param label_key: The key contains ground truth either in the metadata or in the ground_truth_input.
    :param ground_truth_input: The ground_truth_input which should contains data_id_key and label_key.
    :param prediction_data: The path to the prediction data.
    :param perf_data: The path to the perf data.
    :param predict_ground_truth_data: The ground truth data that correspond to the prediction_data.
    :param handle_response_failure: How to handle the response failure.
    :param fallback_value: The fallback value.
    :param is_performance_test: Whether it is a performance test.
    :return: None
    """
    logger.info("Read batch output data now.")
    data_files = [
        f for f in os.listdir(batch_inference_output) if f.endswith("json") or f.endswith("jsonl")
    ]
    logger.info(f"Receiving {data_files}")

    new_df = []
    perf_df = []
    ground_truth = []
    if ground_truth_input:
        input_file_paths = resolve_io_path(ground_truth_input)
        ground_truth_df = pd.DataFrame(read_jsonl_files(input_file_paths))
    else:
        ground_truth_df = None
    if deployment_config_dir:
        online_model = EndpointUtilities.get_model_from_deployment_config_file(deployment_config_dir)
    else:
        online_model = OnlineEndpointModel(None, None, model_type, endpoint_url=endpoint_url)

    rc = ResultConverters(
        online_model._model_type, metadata_key, data_id_key,
        label_key, ground_truth_df, fallback_value=fallback_value, is_performance_test=is_performance_test)
    logger.info("Convert the data now.")
    for f in data_files:
        logger.info(f"Processing file {f}")
        df = pd.read_json(os.path.join(batch_inference_output, f), lines=True)
        for index, row in df.iterrows():
            if not rc.is_result_success(row):
                logger.warn("Met failed response {} at index {} of file {}".format(row, index, f))
                if handle_response_failure == 'neglect':
                    continue
            new_df.append(rc.convert_result(row))
            perf_df.append(rc.convert_result_perf(row))
            if not is_performance_test:
                ground_truth.append(rc.convert_result_ground_truth(row))
            else:
                ground_truth.append({"ground_truth": ''})
    logger.info("Output data now.")
    new_df = pd.DataFrame(new_df)
    perf_df = pd.DataFrame(perf_df)
    ground_truth = pd.DataFrame(ground_truth)
    new_df.to_json(prediction_data, orient="records", lines=True)
    perf_df.to_json(perf_data, orient="records", lines=True)
    ground_truth.to_json(predict_ground_truth_data, orient="records", lines=True)


if __name__ == "__main__":
    args = parse_args()
    main(
        batch_inference_output=args.batch_inference_output,
        model_type=args.model_type,
        data_id_key=args.data_id_key,
        metadata_key=args.metadata_key,
        label_key=args.label_key,
        ground_truth_input=args.ground_truth_input,
        prediction_data=args.prediction_data,
        perf_data=args.perf_data,
        predict_ground_truth_data=args.predict_ground_truth_data,
        handle_response_failure=args.handle_response_failure,
        fallback_value=args.fallback_value,
        is_performance_test=args.is_performance_test,
        endpoint_url=args.endpoint_url,
        deployment_config_dir=args.deployment_config_dir
    )

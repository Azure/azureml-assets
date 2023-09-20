# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Entry script for batch output formatter component."""

import argparse
import os
import pandas as pd

from utils.io import read_pandas_data
from utils.logging import get_logger
from utils.exceptions import swallow_all_exceptions
from .result_converters import ResultConverters


logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse the args for the method."""
    # Input and output arguments
    parser = argparse.ArgumentParser(description=f"{__file__}")
    parser.add_argument("--batch_inference_output", type=str, help="path to prompt crafter output")
    parser.add_argument("--prediction_data", type=str, help="path to output location")
    parser.add_argument("--ground_truth_input", type=str, help="path to output location", default=None)
    parser.add_argument(
        "--predict_ground_truth_data", type=str,
        help="The ground truth data mapping 1-1 to the prediction data.")
    parser.add_argument("--perf_data", type=str, help="path to output location")
    parser.add_argument("--model_type", type=str, help="model type", default='llama')
    parser.add_argument("--metadata_key", type=str, help="metadata key", default=None)
    parser.add_argument("--data_id_key", type=str, help="metadata key", default=None)
    parser.add_argument("--label_key", type=str, help="label key")
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
        predict_ground_truth_data: str
) -> None:
    """Entry script for the script."""
    logger.info("Read batch output data now.")
    data_files = [
        f for f in os.listdir(batch_inference_output) if f.endswith("json") or f.endswith("jsonl")
    ]
    print(f"Receiving {data_files}")

    new_df = []
    perf_df = []
    ground_truth = []
    ground_truth_input = read_pandas_data(ground_truth_input) if ground_truth_input else None
    rc = ResultConverters(
        model_type, metadata_key, data_id_key, label_key, ground_truth_input)
    logger.info("Convert the data now.")
    for f in data_files:
        print(f"Processing file {f}")
        df = pd.read_json(os.path.join(batch_inference_output, f), lines=True)
        for index, row in df.iterrows():
            new_df.append(rc.convert_result(row))
            perf_df.append(rc.convert_result_perf(row))
            ground_truth.append(rc.convert_result_ground_truth(row))
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
        predict_ground_truth_data=args.predict_ground_truth_data
    )

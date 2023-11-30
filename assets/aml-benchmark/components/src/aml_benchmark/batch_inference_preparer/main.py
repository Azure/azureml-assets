# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""The entry script for endpoint input preparer."""

import os
import argparse
import pandas as pd

from aml_benchmark.utils.io import resolve_io_path, read_jsonl_files
from aml_benchmark.utils.logging import get_logger
from aml_benchmark.utils.aml_run_utils import str2bool
from .endpoint_data_preparer import EndpointDataPreparer
from aml_benchmark.utils.exceptions import swallow_all_exceptions
from aml_benchmark.utils.online_endpoint.online_endpoint_model import OnlineEndpointModel


logger = get_logger(__name__)


MLTABLE_CONTENTS = """type: mltable
paths:
  - pattern: ./*.json
transformations:
  - read_json_lines:
      encoding: utf8
      include_path_column: false
"""


def parse_args() -> argparse.Namespace:
    """Parse the args for the method."""
    # Input and output arguments
    parser = argparse.ArgumentParser(description=f"{__file__}")
    parser.add_argument("--input_dataset", type=str, help="Input raw dataset.", default=None)
    parser.add_argument(
        "--batch_input_pattern", nargs='?', const=None, type=str,
        help="The input patterns for the batch endpoint.", default="{}")
    parser.add_argument("--label_key", type=str, help="label key", default=None)
    parser.add_argument("--additional_columns", type=str, help="additional_columns", default=None)
    parser.add_argument(
        "--n_samples", type=int, help="Top samples sending to the endpoint.", default=-1)
    parser.add_argument("--formatted_data", type=str, help="path to output location")
    parser.add_argument("--endpoint_url", type=str, help="endpoint_url", default=None)
    parser.add_argument("--model_type", type=str, help="model_type", default=None)
    parser.add_argument("--output_metadata", type=str, help="path to ground_truth location", default=None)
    parser.add_argument("--is_performance_test", default=False, type=str2bool, help="is_performance_test")
    args, _ = parser.parse_known_args()
    logger.info(f"Arguments: {args}")
    return args


@swallow_all_exceptions(logger)
def main(
    input_dataset: str,
    formatted_dataset: str,
    model_type: str,
    batch_input_pattern: str,
    n_samples: int,
    endpoint_url: str,
    is_performance_test: bool,
    label_key: str,
    additional_columns: str,
    output_metadata: str
) -> None:
    """
    Entry function of the script.

    :param input_dataset: The input dataset which can be MLTable or uri_folder.
    :param formatted_dataset: The output dataset that has the payload.
    :param model_type: The model type.
    :param batch_input_pattern: The input pattern that used to generate payload.
    :param n_samples: The number of samples to generate.
    :param model: The model path.
    :param model_version: The model version.
    :param endpoint: The endpoint url.
    :param is_performance_test: Whether it is performance test.
    :return: None
    """
    online_model = OnlineEndpointModel(None, None, model_type, endpoint_url=endpoint_url)

    endpoint_data_preparer = EndpointDataPreparer(
        online_model._model_type,
        batch_input_pattern,
        label_key=label_key,
        additional_columns=additional_columns)

    # Read the data file into a pandas dataframe
    logger.info("Read data now.")
    if not is_performance_test:
        input_file_paths = resolve_io_path(input_dataset)
    else:
        input_file_paths = [
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources", "perf_data.jsonl")]
    df = pd.DataFrame(read_jsonl_files(input_file_paths))

    # Reformat the columns and save
    new_df = []
    ground_truth_df = []
    sample_count = 0
    logger.info("Process data now.")
    if not is_performance_test:
        for index, row in df.iterrows():
            new_data = endpoint_data_preparer.convert_input_dict(row)
            validate_errors = endpoint_data_preparer.validate_output(new_data)
            ground_truth_data = endpoint_data_preparer.convert_ground_truth(row, new_data)
            if not validate_errors:
                new_df.append(new_data)
                ground_truth_df.append(ground_truth_data)
            else:
                print("Payload {} meeting the following errors: {}.".format(new_data, validate_errors))
            sample_count += 1
            if n_samples > 0 and sample_count > n_samples:
                break
    else:
        new_data = endpoint_data_preparer.convert_input_dict(df.iloc[0])
        ground_truth_df = [
            endpoint_data_preparer.convert_ground_truth(df.iloc[0], new_data)]
        validate_errors = endpoint_data_preparer.validate_output(new_data)
        if n_samples == -1:
            logger.info("n_samples is not entered, use the default value 100.")
            n_samples = 100
        for _ in range(n_samples):
            new_df.append(new_data)

    # Save MLTable file
    logger.info("Writing the data now.")
    with open(os.path.join(formatted_dataset, "MLTable"), "w") as f:
        f.writelines(MLTABLE_CONTENTS)

    new_df = pd.DataFrame(new_df)
    new_df.to_json(os.path.join(formatted_dataset, "formatted_data.json"), orient="records", lines=True)
    ground_truth_df = pd.DataFrame(ground_truth_df)
    ground_truth_df.to_json(
        os.path.join(output_metadata, "ground_truth_data.jsonl"), orient="records", lines=True)


if __name__ == "__main__":
    args = parse_args()
    main(
        input_dataset=args.input_dataset,
        formatted_dataset=args.formatted_data,
        model_type=args.model_type,
        batch_input_pattern=args.batch_input_pattern,
        n_samples=args.n_samples,
        endpoint_url=args.endpoint_url,
        is_performance_test=args.is_performance_test,
        label_key=args.label_key,
        additional_columns=args.additional_columns,
        output_metadata=args.output_metadata
    )

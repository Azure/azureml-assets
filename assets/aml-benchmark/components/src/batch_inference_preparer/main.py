# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""The entry script for endpoint input preparer."""

import os
import argparse
import pandas as pd

from utils.io import read_pandas_data
from utils.logging import get_logger
from .endpoint_data_preparer import EndpointDataPreparer
from utils.exceptions import swallow_all_exceptions


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
    parser.add_argument("--input_dataset", type=str, help="Input raw dataset.")
    parser.add_argument(
        "--batch_input_pattern", nargs='?', const=None, type=str,
        help="The input patterns for the batch endpoint.", default="{}")
    parser.add_argument(
        "--model_type", type=str,
        help="The end point model type, possible values are `llama`, `aoai`.", default='llama')
    parser.add_argument(
        "--n_samples", type=int, help="Top samples sending to the endpoint.", default=-1)
    parser.add_argument("--formatted_data", type=str, help="path to output location")
    parser.add_argument("--model_asset_path", type=str, help="model_asset_path", default=None)
    parser.add_argument("--deployment_name", type=str, help="deployment_name", default=None)
    parser.add_argument("--endpoint_name", type=str, help="endpoint_name", default=None)
    parser.add_argument("--vm_sku", type=str, help="vm_sku", default=None)
    args, _ = parser.parse_known_args()
    logger.info(f"Arguments: {args}")
    return args


@swallow_all_exceptions(logger)
def main(
    input_dataset: str,
    formatted_dataset: str,
    model_type: str,
    batch_input_pattern: str
) -> None:
    """
    Entry function of the script.

    :param input_dataset: The input dataset which can be MLTable or uri_folder.
    :param formatted_dataset: The output dataset that has the payload.
    :param model_type: The model type.
    :param batch_input_pattern: The input pattern that used to generate payload.
    :return: None
    """
    # online_endpoint = OnlineEndpoint.from_args(args)
    # if online_endpoint.should_deploy():
    #     online_endpoint.deploy_model()

    endpoint_data_preparer = EndpointDataPreparer(model_type, batch_input_pattern)

    # Read the data file into a pandas dataframe
    logger.info("Read data now.")
    df = read_pandas_data(input_dataset)

    # Reformat the columns and save
    new_df = []
    sample_count = 0
    logger.info("Process data now.")
    for index, row in df.iterrows():
        new_data = endpoint_data_preparer.convert_input_dict(row)
        validate_errors = endpoint_data_preparer.validate_output(new_data)
        if not validate_errors:
            new_df.append(new_data)
        else:
            print("Payload {} meeting the following errors: {}.".format(new_data, validate_errors))
        sample_count += 1
        if args.n_samples > 0 and sample_count > args.n_samples:
            break

    # Save MLTable file
    logger.info("Writing the data now.")
    with open(os.path.join(formatted_dataset, "MLTable"), "w") as f:
        f.writelines(MLTABLE_CONTENTS)

    new_df = pd.DataFrame(new_df)
    new_df.to_json(os.path.join(formatted_dataset, "formatted_data.json"), orient="records", lines=True)


if __name__ == "__main__":
    args = parse_args()
    main(
        input_dataset=args.input_dataset,
        formatted_dataset=args.formatted_data,
        model_type=args.model_type,
        batch_input_pattern=args.batch_input_pattern
    )

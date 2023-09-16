# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
import os
import argparse
import pandas as pd

from utils.io import read_pandas_data
from utils.logging import get_logger
from .endpoint_data_preparer import EndpointDataPreparer


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
    parser.add_argument("--batch_input_pattern", nargs='?', const=None, type=str, help="The input patterns for the batch endpoint.", default="{}")
    parser.add_argument("--model_type", type=str, help="The end point model type, possible values are `llama`, `aoai`.", default='llama')
    parser.add_argument("--n_samples", type=int, help="Top samples sending to the endpoint.", default=-1)
    parser.add_argument("--formatted_data", type=str, help="path to output location")
    parser.add_argument("--model_asset_path", type=str, help="model_asset_path", default=None)
    parser.add_argument("--deployment_name", type=str, help="deployment_name", default=None)
    parser.add_argument("--endpoint_name", type=str, help="endpoint_name", default=None)
    parser.add_argument("--vm_sku", type=str, help="vm_sku", default=None)
    args, _ = parser.parse_known_args()
    logger.info(f"Arguments: {args}")
    return args


def main():
    """Main function of the script."""   
    args = parse_args()

    input_dataset = args.input_dataset
    formatted_dataset = args.formatted_data

    # online_endpoint = OnlineEndpoint.from_args(args)
    # if online_endpoint.should_deploy():
    #     online_endpoint.deploy_model()
    
    endpoint_data_preparer = EndpointDataPreparer.from_args(args)

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
    with open(os.path.join(formatted_dataset, "MLTable"),"w+") as f:
        f.writelines(MLTABLE_CONTENTS)

    new_df = pd.DataFrame(new_df)
    new_df.to_json(os.path.join(formatted_dataset, "formatted_data.json"), orient="records", lines=True)


if __name__ == "__main__":
    main()

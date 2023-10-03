# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""The entry script for endpoint input preparer."""

import os
import argparse
import pandas as pd

from utils.io import resolve_io_path, read_jsonl_files
from utils.logging import get_logger
from utils.aml_run_utils import str2bool
from .endpoint_data_preparer import EndpointDataPreparer
from utils.exceptions import swallow_all_exceptions
from utils.online_endpoint.online_endpoint import OnlineEndpoint, ResourceState
from utils.online_endpoint.online_endpoint_factory import OnlineEndpointFactory
from utils.online_endpoint.online_endpoint_model import OnlineEndpointModel
from utils.online_endpoint.endpoint_utils import EndpointUtilities


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
    parser.add_argument(
        "--model_type", type=str,
        help="The end point model type, possible values are `oss`, `aoai`.", default='oss')
    parser.add_argument(
        "--n_samples", type=int, help="Top samples sending to the endpoint.", default=-1)
    parser.add_argument("--formatted_data", type=str, help="path to output location")
    parser.add_argument("--deployment_name", type=str, help="deployment_name", default=None)
    parser.add_argument("--endpoint", type=str, help="endpoint", default=None)
    parser.add_argument("--deployment_sku", type=str, help="deployment_sku", default=None)
    parser.add_argument("--model", type=str, help="model", default=None)
    parser.add_argument("--model_version", type=str, help="model_version", default=None)
    parser.add_argument("--endpoint_workspace", type=str, help="endpoint_workspace", default=None)
    parser.add_argument("--endpoint_resource_group", type=str, help="endpoint_resource_group", default=None)
    parser.add_argument("--endpoint_subscription_id", type=str, help="endpoint_subscription_id", default=None)
    parser.add_argument("--endpoint_location", type=str, help="endpoint_location", default=None)
    parser.add_argument("--output_metadata", type=str, help="output_metadata", default=None)
    parser.add_argument("--is_performance_test", default=False, type=str2bool, help="is_performance_test")
    args, _ = parser.parse_known_args()
    logger.info(f"Arguments: {args}")
    return args


def deploy_model_maybe(online_endpoint: OnlineEndpoint, output_metadata_dir: str) -> None:
    """Deploy the model if it is not deployed."""
    managed_endpoint = False
    managed_deployment = False
    if online_endpoint.endpoint_state() == ResourceState.FAILURE:
        logger.info("Endpoint is in failure state, delete it now.")
        online_endpoint.delete_endpoint()
    if online_endpoint.endpoint_state() == ResourceState.NOT_FOUND:
        logger.info("Endpoint is not found, create it now.")
        online_endpoint.create_endpoint()
        managed_endpoint = True
    if online_endpoint.deployment_state() == ResourceState.FAILURE:
        logger.info("Deployment is in failure state, delete it now.")
        online_endpoint.delete_deployment()
    if online_endpoint.deployment_state() == ResourceState.NOT_FOUND:
        logger.info("Deployment is not found, create it now.")
        online_endpoint.create_deployment()
        managed_deployment = True
    EndpointUtilities.dump_endpoint_metadata_json(
        online_endpoint, managed_endpoint, managed_deployment, output_metadata_dir)
    print("Endpoint is ready/checked now.")


@swallow_all_exceptions(logger)
def main(
    input_dataset: str,
    formatted_dataset: str,
    model_type: str,
    batch_input_pattern: str,
    n_samples: int,
    model: str,
    model_version: str,
    endpoint: str,
    endpoint_workspace: str,
    endpoint_resource_group: str,
    endpoint_subscription_id: str,
    endpoint_location: str,
    deployment_name: str,
    deployment_sku: str,
    output_metadata_dir: str,
    is_performance_test: bool
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
    :param endpoint_workspace: The endpoint workspace.
    :param endpoint_resource_group: The endpoint resource group.
    :param endpoint_subscription_id: The endpoint subscription id.
    :param endpoint_location: The endpoint location.
    :param deployment_name: The deployment name.
    :param deployment_sku: The deployment sku.
    :param output_metadata_dir: The output metadata dir.
    :param is_performance_test: Whether it is performance test.
    :return: None
    """
    online_endpoint = OnlineEndpointFactory.get_online_endpoint(
        endpoint_workspace,
        endpoint_resource_group,
        endpoint_subscription_id,
        OnlineEndpointModel(
            model,
            model_version,
            model_type
        ),
        endpoint,
        deployment_name,
        deployment_sku,
        endpoint_location
    )
    deploy_model_maybe(online_endpoint, output_metadata_dir)

    endpoint_data_preparer = EndpointDataPreparer(model_type, batch_input_pattern)

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
    sample_count = 0
    logger.info("Process data now.")
    if not is_performance_test:
        for index, row in df.iterrows():
            new_data = endpoint_data_preparer.convert_input_dict(row)
            validate_errors = endpoint_data_preparer.validate_output(new_data)
            if not validate_errors:
                new_df.append(new_data)
            else:
                print("Payload {} meeting the following errors: {}.".format(new_data, validate_errors))
            sample_count += 1
            if n_samples > 0 and sample_count > n_samples:
                break
    else:
        new_data = endpoint_data_preparer.convert_input_dict(df.iloc[0])
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


if __name__ == "__main__":
    args = parse_args()
    main(
        input_dataset=args.input_dataset,
        formatted_dataset=args.formatted_data,
        model_type=args.model_type,
        batch_input_pattern=args.batch_input_pattern,
        n_samples=args.n_samples,
        model=args.model,
        model_version=args.model_version,
        endpoint=args.endpoint,
        endpoint_workspace=args.endpoint_workspace,
        endpoint_resource_group=args.endpoint_resource_group,
        endpoint_subscription_id=args.endpoint_subscription_id,
        endpoint_location=args.endpoint_location,
        deployment_name=args.deployment_name,
        deployment_sku=args.deployment_sku,
        output_metadata_dir=args.output_metadata,
        is_performance_test=args.is_performance_test
    )

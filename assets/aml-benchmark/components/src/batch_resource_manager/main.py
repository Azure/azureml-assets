# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# ---------------------------------------------------------

"""Entry script for Batch resource manager."""

import argparse

from utils.logging import get_logger
from utils.exceptions import swallow_all_exceptions
from utils.aml_run_utils import str2bool
from utils.online_endpoint.endpoint_utils import EndpointUtilities
from utils.online_endpoint.online_endpoint_factory import OnlineEndpointFactory
from utils.online_endpoint.online_endpoint import OnlineEndpoint, ResourceState
from utils.online_endpoint.online_endpoint_model import OnlineEndpointModel


logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse the args for the method."""
    # Input and output arguments
    parser = argparse.ArgumentParser(description=f"{__file__}")
    parser.add_argument("--model_type", type=str, help="model type", default='llama')
    parser.add_argument(
        "--delete_managed_resources",
        default=False, type=str2bool,
        help="Delete managed resources create during the run.",
    )
    parser.add_argument(
        "--deployment_metadata_dir", default=None, type=str,
        help="Directory contains deployment metadata.",
    )
    parser.add_argument("--deployment_name", type=str, help="deployment_name", default=None)
    parser.add_argument("--endpoint_name", type=str, help="endpoint_name", default=None)
    parser.add_argument("--deployment_sku", type=str, help="deployment_sku", default=None)
    parser.add_argument("--model", type=str, help="model", default=None)
    parser.add_argument("--model_version", type=str, help="model_version", default=None)
    parser.add_argument("--endpoint_workspace", type=str, help="endpoint_workspace", default=None)
    parser.add_argument("--endpoint_resource_group", type=str, help="endpoint_resource_group", default=None)
    parser.add_argument("--endpoint_subscription_id", type=str, help="endpoint_subscription_id", default=None)
    parser.add_argument("--endpoint_location", type=str, help="endpoint_location", default=None)
    parser.add_argument("--output_metadata", type=str, help="output_metadata", default=None)
    parser.add_argument("--connections_name", type=str, help="output_metadata", default=None)
    args, _ = parser.parse_known_args()
    logger.info(f"Arguments: {args}")
    return args


def delete_managed_resources_maybe(
        output_metadata_dir: bool,
        deployment_metadata: dict,
        online_endpoint: OnlineEndpoint
) -> None:
    """Delete managed resources if delete_managed_resources is True."""
    logger.info("Deleting managed resources.")
    is_endpoint_deleted = False
    is_deployment_deleted = False
    is_connections_deleted = False
    if deployment_metadata.get('is_managed_endpoint', True):
        logger.info("Deleting endpoint.")
        online_endpoint.delete_endpoint()
        is_endpoint_deleted = True
        is_deployment_deleted = True
    elif deployment_metadata.get('is_managed_deployment', True):
        logger.info("Deleting deployment.")
        online_endpoint.delete_deployment()
        is_deployment_deleted = True
    if deployment_metadata.get('is_managed_connections', True):
        logger.info("Deleting connections.")
        online_endpoint.delete_connections()
        is_connections_deleted = True
    EndpointUtilities.dump_delete_status(
        is_endpoint_deleted,
        is_deployment_deleted,
        is_connections_deleted,
        output_metadata_dir
    )


def deploy_model_maybe(online_endpoint: OnlineEndpoint, output_metadata_dir: str) -> None:
    """Deploy the model if it is not deployed."""
    managed_endpoint = False
    managed_deployment = False
    managed_connections = True
    logger.info(
        f'input endpoint state: {online_endpoint.endpoint_state()}, '
        f'deployment state: {online_endpoint.deployment_state()}')
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
    online_endpoint.create_connections()
    EndpointUtilities.dump_endpoint_metadata_json(
        online_endpoint, managed_endpoint, managed_deployment, managed_connections, output_metadata_dir)
    logger.info("Endpoint is ready/checked now.")


@swallow_all_exceptions(logger)
def main(
    model: str,
    model_version: str,
    model_type: str,
    endpoint_name: str,
    endpoint_workspace: str,
    endpoint_resource_group: str,
    endpoint_subscription_id: str,
    endpoint_location: str,
    deployment_name: str,
    deployment_sku: str,
    output_metadata_dir: str,
    deployment_metadata_dir: str,
    delete_managed_resources: bool,
    connections_name: str
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
    if not delete_managed_resources:
        online_model = OnlineEndpointModel(model, model_version, model_type, endpoint_name)
        online_endpoint = OnlineEndpointFactory.get_online_endpoint(
            endpoint_workspace,
            endpoint_resource_group,
            endpoint_subscription_id,
            online_model,
            endpoint_name,
            deployment_name,
            deployment_sku,
            endpoint_location,
            connections_name=connections_name
        )
        deploy_model_maybe(online_endpoint, output_metadata_dir)
    else:
        if not deployment_metadata_dir:
            online_model = OnlineEndpointModel(model, model_version, model_type, endpoint_name)
            online_endpoint = OnlineEndpointFactory.get_online_endpoint(
                endpoint_workspace,
                endpoint_resource_group,
                endpoint_subscription_id,
                online_model,
                endpoint_name,
                deployment_name,
                deployment_sku,
                endpoint_location,
                connections_name=connections_name
            )
            deployment_metadata = {}
        else:
            deployment_metadata = EndpointUtilities.load_endpoint_metadata_json(deployment_metadata_dir)
            online_endpoint = OnlineEndpointFactory.from_metadata(deployment_metadata)
        delete_managed_resources_maybe(output_metadata_dir, deployment_metadata, online_endpoint)


if __name__ == "__main__":
    args = parse_args()
    main(
        args.model,
        args.model_version,
        args.model_type,
        args.endpoint_name,
        args.endpoint_workspace,
        args.endpoint_resource_group,
        args.endpoint_subscription_id,
        args.endpoint_location,
        args.deployment_name,
        args.deployment_sku,
        args.output_metadata,
        args.deployment_metadata_dir,
        args.delete_managed_resources,
        args.connections_name
    )

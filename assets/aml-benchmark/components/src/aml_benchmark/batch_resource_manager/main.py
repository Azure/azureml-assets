# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# ---------------------------------------------------------

"""Entry script for Batch resource manager."""

from typing import List, Generator, Optional, Tuple
import argparse
import json
import subprocess
import sys
import time
import traceback

from azureml.core import Run
from aml_benchmark.utils.logging import get_logger, BufferStore
from aml_benchmark.utils.exceptions import swallow_all_exceptions
from aml_benchmark.utils.aml_run_utils import str2bool, get_dependent_run
from aml_benchmark.utils.online_endpoint.endpoint_utils import EndpointUtilities
from aml_benchmark.utils.online_endpoint.online_endpoint_factory import OnlineEndpointFactory
from aml_benchmark.utils.online_endpoint.online_endpoint import OnlineEndpoint, ResourceState
from aml_benchmark.utils.online_endpoint.online_endpoint_model import OnlineEndpointModel
from aml_benchmark.utils.exceptions import BenchmarkUserException
from aml_benchmark.utils.error_definitions import BenchmarkUserError
from azureml._common._error_definition.azureml_error import AzureMLError
from azureml._restclient.constants import RunStatus


logger = get_logger(__name__)

# TODO: check back here if there is a way to have a published contract supporitng this
_FINAL_STATUSES = { RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELED }

def parse_args() -> argparse.Namespace:
    """Parse the args for the method."""
    # Input and output arguments
    parser = argparse.ArgumentParser(description=f"{__file__}")
    parser.add_argument("--model_type", type=str, help="model type", default='oss')
    parser.add_argument(
        "--deletion_model",
        default=True, type=str2bool,
        help="Delete managed resources create during the run.",
    )
    parser.add_argument(
        "--deployment_metadata", default=None, type=str,
        help="Directory contains deployment metadata.",
    )
    parser.add_argument("--cli_file", type=str, help="cli_file", default=None)
    parser.add_argument("--deployment_name", type=str, help="deployment_name", default=None)
    parser.add_argument("--endpoint_name", type=str, help="endpoint_name", default=None)
    parser.add_argument("--deployment_sku", type=str, help="deployment_sku", default=None)
    parser.add_argument("--deployment_env", type=str, help="deployment_env", default=None)
    parser.add_argument("--model", type=str, help="model", default=None)
    parser.add_argument("--model_version", type=str, help="model_version", default=None)
    parser.add_argument("--endpoint_workspace", type=str, help="endpoint_workspace", default=None)
    parser.add_argument("--endpoint_resource_group", type=str, help="endpoint_resource_group", default=None)
    parser.add_argument("--endpoint_subscription_id", type=str, help="endpoint_subscription_id", default=None)
    parser.add_argument("--endpoint_location", type=str, help="endpoint_location", default=None)
    parser.add_argument("--output_metadata", type=str, help="output_metadata", default=None)
    parser.add_argument("--connections_name", type=str, help="output_metadata", default=None)
    parser.add_argument("--additional_deployment_env_vars", type=str, default=None)
    parser.add_argument("--finetuned_workspace", type=str, help="finetuned_workspace", default=None)
    parser.add_argument("--finetuned_resource_group", type=str, help="finetuned_resource_group", default=None)
    parser.add_argument("--finetuned_subscription_id", type=str, help="finetuned_subscription_id", default=None)
    parser.add_argument(
        "--do_quota_validation",
        default=True, type=str2bool,
        help="If doing quota valiation or not for AOAI model.",
    )
    parser.add_argument(
        "--redeploy_model",
        default=False, type=str2bool,
        help="Force redeploy models or not.",
    )
    parser.add_argument(
        "--wait_finetuned_step",
        default=False, type=str2bool,
        help="Force redeploymodels or not.",
    )
    parser.add_argument(
        "--finetuned_start_timeout_seconds",
        default=900, 
        type=int,
        help="Maximum time to wait for fine tune step to start.",
    )
    parser.add_argument(
        "--finetuned_step_name", type=str, help="finetuned_step_name", default=None)
    parser.add_argument(
        "--delete_managed_deployment",
        default=False, type=str2bool,
        help="If deleting the managed deployment.",
    )
    parser.add_argument(
        "--use_max_quota",
        default=True, type=str2bool,
        help="If using max quota or not for AOAI model.",
    )
    parser.add_argument(
        "--is_finetuned_model",
        default=False, type=str2bool,
        help="Flag on if this is a finetuned model.",
    )
    parser.add_argument(
        "--finetuned_model_metadata", default=None, type=str,
        help="Directory contains finetuned_model_metadata.",
    )
    parser.add_argument("--deployment_retries", default=5, type=int, help="Number of retries for deployment.")
    parser.add_argument(
        "--deployment_retry_interval_seconds", default=600, type=int, help="Retry interval for deployment.")
    args, _ = parser.parse_known_args()
    logger.info(f"Arguments: {args}")
    return args


def delete_managed_resources_maybe(
        output_metadata_dir: bool,
        deployment_metadata_dict: dict,
        online_endpoint: OnlineEndpoint
) -> None:
    """Delete managed resources if delete_managed_resources is True."""
    logger.info("Deleting managed resources.")
    is_endpoint_deleted = False
    is_deployment_deleted = False
    is_connections_deleted = False
    # For some OAI model, the deployment needs to be delete
    if deployment_metadata_dict.get('is_managed_deployment', True):
        logger.info("Deleting deployment.")
        try:
            online_endpoint.delete_deployment()
            is_deployment_deleted = True
        except Exception as e:
            logger.error(f"Failed to delete deployment with error {e}.")
            logger.error(traceback.format_exception(*sys.exc_info()))
    if deployment_metadata_dict.get('is_managed_connections', True):
        logger.info("Deleting connections.")
        try:
            online_endpoint.delete_connections()
            is_connections_deleted = True
        except Exception as e:
            logger.error(f"Failed to delete connections with error {e}.")
            logger.error(traceback.format_exception(*sys.exc_info()))
    if deployment_metadata_dict.get('is_managed_endpoint', True):
        logger.info("Deleting endpoint.")
        try:
            online_endpoint.delete_endpoint()
            is_endpoint_deleted = True
        except Exception as e:
            logger.error(f"Failed to delete endpoint with error {e}.")
            logger.error(traceback.format_exception(*sys.exc_info()))
    EndpointUtilities.dump_delete_status(
        is_endpoint_deleted,
        is_deployment_deleted,
        is_connections_deleted,
        output_metadata_dir
    )


def deploy_endpoint_maybe(online_endpoint: OnlineEndpoint) -> bool:
    """Deploy the endpoint if it is not deployed."""
    managed_endpoint = False
    logger.info(
        f'input endpoint state: {online_endpoint.endpoint_state()} in '
        f'{online_endpoint.subscription_id} and {online_endpoint.resource_group}.')
    if online_endpoint.endpoint_state() == ResourceState.FAILURE:
        logger.info("Endpoint is in failure state, delete it now.")
        online_endpoint.delete_endpoint()
    if online_endpoint.endpoint_state() == ResourceState.NOT_FOUND:
        logger.info("Endpoint is not found, create it now.")
        online_endpoint.create_endpoint()
        managed_endpoint = True
    logger.info("endpoint is ready/checked now.")
    return managed_endpoint


def deploy_model_maybe(
        online_endpoint: OnlineEndpoint, output_metadata_dir: str,
        managed_endpoint: bool, redeploy_model: bool
) -> Tuple[bool, bool]:
    """Deploy the model if it is not deployed."""
    managed_deployment = False
    managed_connections = True
    logger.info(
        f'deployment state: {online_endpoint.deployment_state()} in '
        f'{online_endpoint.subscription_id} and {online_endpoint.resource_group}.')

    if online_endpoint.deployment_state() == ResourceState.FAILURE:
        logger.info("Deployment is in failure state, delete it now.")
        online_endpoint.delete_deployment()
    if online_endpoint.deployment_state() == ResourceState.NOT_FOUND or redeploy_model:
        logger.info("Deployment is not found, create it now.")
        online_endpoint.create_deployment()
        managed_deployment = True
    is_conn_created = online_endpoint.create_connections()
    EndpointUtilities.dump_endpoint_metadata_json(
        online_endpoint, managed_endpoint, managed_deployment, managed_connections, output_metadata_dir)
    if not is_conn_created:
        logger.warning("Connections were not created. The deployment may not work properly.")
    logger.info("model is ready/checked now.")
    return managed_deployment, is_conn_created


def _online_endpoint_generator(
        subscription_lists: List[str],
        region_lists: List[str],
        online_model: OnlineEndpointModel,
        endpoint_name: str,
        endpoint_workspace: str,
        endpoint_resource_group: str,
        deployment_name: str,
        deployment_sku: str,
        connections_name: str,
        do_quota_validation: bool,
        additional_deployment_env_vars: str,
        use_max_quota: bool,
        deployment_env: str
) -> Generator[OnlineEndpoint, None, None]:
    if deployment_sku is None:
        deployment_sku = 1
        use_max_quota = True
    else:
        deployment_sku = int(deployment_sku)
    for subscription in subscription_lists:
        for region in region_lists:
            if region is None:
                logger.info("No region received, using workspace region.")
                region = Run.get_context().experiment.workspace.location
            logger.info("Checking {} in region {}.".format(subscription, region))
            online_endpoint = OnlineEndpointFactory.get_online_endpoint(
                endpoint_workspace,
                endpoint_resource_group,
                subscription,
                online_model,
                endpoint_name,
                deployment_name,
                deployment_sku,
                region,
                connections_name=connections_name,
                additional_deployment_env_vars=json.loads(
                    additional_deployment_env_vars) if additional_deployment_env_vars else {},
                deployment_env=deployment_env
            )
            try:
                if do_quota_validation and online_model.is_aoai_model():
                    current_quota = online_endpoint.model_quota()
                    logger.info(
                        "Current quota for model {} is {}.".format(
                            online_model.model_name, current_quota))
                    if current_quota < deployment_sku:
                        logger.warning(
                            f"For {region}, current quota is less than sku {deployment_sku}, skip it now.")
                        continue
                    if use_max_quota:
                        if online_model.is_finetuned:
                            current_quota = min(current_quota, 50)
                        online_endpoint._sku = current_quota
            except Exception as e:
                logger.warning(f"Failed to get quota with error {e}.")
                logger.warning(
                    "To leverage quota validation, please make sure the "
                    "managed identity has contributor role on the resource group.")
                logger.warning(traceback.format_exception(*sys.exc_info()))

            yield online_endpoint


def deploy_model_in_list_maybe(
        subscriptions_list: List[str],
        locations_list: List[str],
        online_model: OnlineEndpointModel,
        endpoint_name: str,
        endpoint_workspace: str,
        endpoint_resource_group: str,
        deployment_name: str,
        deployment_sku: str,
        output_metadata_dir: str,
        connections_name: str,
        do_quota_validation: bool,
        additional_deployment_env_vars: str,
        use_max_quota: bool,
        deployment_env: str,
        redeploy_model: bool
) -> bool:
    """Deploy the model using different regions."""
    for online_endpoint in _online_endpoint_generator(
                subscriptions_list,
                locations_list,
                online_model,
                endpoint_name,
                endpoint_workspace,
                endpoint_resource_group,
                deployment_name,
                deployment_sku,
                connections_name,
                do_quota_validation,
                additional_deployment_env_vars,
                use_max_quota,
                deployment_env
    ):
        managed_endpoint = False
        try:
            managed_endpoint = deploy_endpoint_maybe(online_endpoint)
        except Exception as e:
            logger.error(f"Failed to deploy endpoint with error {e}.")
            continue
        managed_deployment = False
        try:
            managed_deployment, is_conn_created = deploy_model_maybe(
                online_endpoint, output_metadata_dir, managed_endpoint, redeploy_model)
            if not is_conn_created:
                error_msg = "Deployment was created, but connection was not, making it unusable."
                raise ValueError(error_msg)
            logger.info("Deployment and connections are created successfully.")
            return managed_deployment
        except Exception as e:
            logger.error(f"Model deployment failed with error {e}.")
            logger.error(traceback.format_exception(*sys.exc_info()))
        # If endpoint is deployed successfully, the deletion code won't be reached
        if managed_deployment and online_endpoint.deployment_state() != ResourceState.NOT_FOUND:
            logger.info("Deleting failed deployment now.")
            online_endpoint.delete_deployment()
        if managed_endpoint:
            logger.info("Deleting created endpoint now.")
            online_endpoint.delete_endpoint()
    return False


# the function must not return None on a non-error path
def _repeat_till_success_or_timeout(timeout_seconds : int, interval_seconds : int, func, *args, **kwargs):
    start_time = time.time()
    while not (result:=func(*args, **kwargs)):
        if time.time() - start_time > timeout_seconds:
            return None
        logger.info("Waiting for %d seconds before retrying to get dependent run.", interval_seconds)
        time.sleep(interval_seconds)
    return result


def _wait_finetuned_step(finetuned_step_name: Optional[str], start_timeout_seconds = 900) -> None:
    """Wait for finetuned step to finish."""
    finetuned_step_name = finetuned_step_name if finetuned_step_name else "openai_completions_finetune_pipeline"
    logger.info(f"Wait for finetuned step {finetuned_step_name} to finish.")
    wait_step_run = _repeat_till_success_or_timeout(start_timeout_seconds,30,get_dependent_run,finetuned_step_name)
    if wait_step_run is None:
        logger.error(f"Finetuned wait step {finetuned_step_name} not found or failed to "
                     f"start in allotted {start_timeout_seconds} seconds.")
        raise BenchmarkUserException._with_error(
            AzureMLError.create(
                BenchmarkUserError,
                error_details=f"Finetuned wait step {finetuned_step_name} not found or failed to start in allotted {start_timeout_seconds} seconds.")
            )
    # TODO: this needs a timeout    
    while (wait_step_status:=wait_step_run.get_status()) not in _FINAL_STATUSES:
        logger.info("Waiting for finetuned step to finish. Current status is %s.", wait_step_status)
        time.sleep(60)
    if wait_step_status != RunStatus.COMPLETED:
        logger.error(f"Finetuned wait step {finetuned_step_name} is not COMPLETED, status: %s.", wait_step_run.get_status())
        raise BenchmarkUserException._with_error(
            AzureMLError.create(
                BenchmarkUserError,
                error_details=f"Finetuned wait step {finetuned_step_name} is not successful, status: {wait_step_status}.")
            )

    logger.info(f"Finetuned wait step {finetuned_step_name} is COMPLETED.")


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
    deployment_metadata: str,
    deletion_model: bool,
    connections_name: str,
    additional_deployment_env_vars: str,
    do_quota_validation: bool,
    use_max_quota: bool,
    redeploy_model: bool,
    deployment_env: str,
    cli_file: str,
    is_finetuned_model: bool,
    finetuned_subscription_id: str,
    finetuned_resource_group: str,
    finetuned_workspace: str,
    delete_managed_deployment: bool,
    deployment_retries: int,
    deployment_retry_interval_seconds: int,
    wait_finetuned_step: bool,
    finetuned_step_name: str,
    finetuned_start_timeout_seconds: int
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
    if cli_file:
        subprocess.run(
            [
                "az ml online-deployment create -f", cli_file, "--workspace-name",
                endpoint_workspace, "--resource-group", endpoint_resource_group,
                "--subscription", endpoint_subscription_id
            ], check=True
        )
    if not deletion_model:
        if wait_finetuned_step:
            _wait_finetuned_step(finetuned_step_name, finetuned_start_timeout_seconds)
        subscriptions_list = [
            s.strip() for s in endpoint_subscription_id.split(',')] if endpoint_subscription_id else [None]
        locations_list = [
            s.strip() for s in endpoint_location.split(',')] if endpoint_location else [None]
        if is_finetuned_model:
            logger.info("Detected a finetuned model, setting up an online endpoint.")
            workspace = Run.get_context().experiment.workspace
            finetuned_workspace = finetuned_workspace if finetuned_workspace else workspace.name
            finetuned_resource_group = finetuned_resource_group \
                if finetuned_resource_group else workspace.resource_group
            finetuned_subscription_id = finetuned_subscription_id \
                if finetuned_subscription_id else workspace.subscription_id

        # if model is finetuned from aoai finetune step then override model from finetuned_model_metadata
        is_aoai_finetuned_model = finetuned_model_metadata is not None
        if is_aoai_finetuned_model:
            with open(finetuned_model_metadata) as f:
                model_metadata = json.load(f)
                model = model_metadata["finetuned_model_id"]

        online_model = OnlineEndpointModel(
            model, model_version, model_type, endpoint_name, is_finetuned_model,
            finetuned_subscription_id, finetuned_resource_group, finetuned_workspace,
            finetuned_step_name, is_aoai_finetuned_model
        )
        is_deployment_successful = False
        if deployment_retries <= 0:
            logger.warning("Deployment retries is less than 0, resetting it to 1 to allow at least one attempt")
            deployment_retries = 1
        retries_err_msg = f"Deployment is not successful after {deployment_retries} retries."
        while deployment_retries > 0:
            is_deployment_successful = deploy_model_in_list_maybe(
                subscriptions_list,
                locations_list,
                online_model,
                endpoint_name,
                endpoint_workspace,
                endpoint_resource_group,
                deployment_name,
                deployment_sku,
                output_metadata_dir,
                connections_name,
                do_quota_validation,
                additional_deployment_env_vars,
                use_max_quota,
                deployment_env,
                redeploy_model
            )
            if is_deployment_successful:
                logger.info("Deployment is successful.")
                break
            else:
                logger.warning("Deployment is not successful, retrying now.")
                deployment_retries -= 1
                time.sleep(deployment_retry_interval_seconds)
        if not is_deployment_successful:
            raise BenchmarkUserException._with_error(
                AzureMLError.create(
                    BenchmarkUserError,
                    error_details=f"{retries_err_msg} Details: {BufferStore.get_all_data()}"
                ))
    elif delete_managed_deployment:
        if not deployment_metadata:
            logger.info("Delete deployment using input parameters.")
            is_aoai_finetuned_model = finetuned_model_metadata is not None
            online_model = OnlineEndpointModel(model, model_version, model_type, endpoint_name,
                                               is_aoai_finetuned_model=is_aoai_finetuned_model)
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
            deployment_metadata_dict = {}
        else:
            deployment_metadata_dict = EndpointUtilities.load_endpoint_metadata_json(deployment_metadata)
            logger.info("Loaded deployment metadata {}.".format(deployment_metadata_dict))
            online_endpoint = OnlineEndpointFactory.from_metadata(deployment_metadata_dict)
        delete_managed_resources_maybe(output_metadata_dir, deployment_metadata_dict, online_endpoint)
    else:
        logger.info("Skip deletion as delete_managed_deployment is False.")


if __name__ == "__main__":
    args = parse_args()
    main(
        model=args.model,
        model_version=args.model_version,
        model_type=args.model_type,
        endpoint_name=args.endpoint_name,
        endpoint_workspace=args.endpoint_workspace,
        endpoint_resource_group=args.endpoint_resource_group,
        endpoint_subscription_id=args.endpoint_subscription_id,
        endpoint_location=args.endpoint_location,
        deployment_name=args.deployment_name,
        deployment_sku=args.deployment_sku,
        output_metadata_dir=args.output_metadata,
        deployment_metadata=args.deployment_metadata,
        deletion_model=args.deletion_model,
        connections_name=args.connections_name,
        additional_deployment_env_vars=args.additional_deployment_env_vars,
        do_quota_validation=args.do_quota_validation,
        use_max_quota=args.use_max_quota,
        redeploy_model=args.redeploy_model,
        deployment_env=args.deployment_env,
        cli_file=args.cli_file,
        is_finetuned_model=args.is_finetuned_model,
        finetuned_subscription_id=args.finetuned_subscription_id,
        finetuned_resource_group=args.finetuned_resource_group,
        finetuned_workspace=args.finetuned_workspace,
        delete_managed_deployment=args.delete_managed_deployment,
        deployment_retries=args.deployment_retries,
        deployment_retry_interval_seconds=args.deployment_retry_interval_seconds,
        wait_finetuned_step=args.wait_finetuned_step,
        finetuned_step_name=args.finetuned_step_name,
        finetuned_start_timeout_seconds=args.finetuned_start_timeout_seconds
    )

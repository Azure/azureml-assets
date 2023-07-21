# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Run Model deployment module."""
import argparse
import json
import re
import time

from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    OnlineRequestSettings,
    ProbeSettings,
)
from azureml._common._error_definition import AzureMLError
from azureml._common.exceptions import AzureMLException
from pathlib import Path

from utils.config import AppName, ComponentVariables
from utils.common_utils import get_mlclient, get_model_name
from utils.logging_utils import custom_dimensions, get_logger
from utils.exceptions import (
    swallow_all_exceptions,
    OnlineEndpointInvocationError,
    EndpointCreationError,
    DeploymentCreationError,
)


MAX_REQUEST_TIMEOUT = 90000
MAX_INSTANCE_COUNT = 20
MAX_DEPLOYMENT_LOG_TAIL_LINES = 10000

logger = get_logger(__name__)
custom_dimensions.app_name = AppName.DEPLOY_MODEL


def parse_args():
    """Return arguments."""
    parser = argparse.ArgumentParser()

    # Defaults for managed online endpoint has been picked mostly from:
    # https://learn.microsoft.com/en-us/azure/machine-learning/reference-yaml-deployment-managed-online
    # Some of the defaults have been tweaked to cater to large models.

    # add arguments
    parser.add_argument(
        "--registration_details_folder",
        type=Path,
        help="Folder containing model registration details in a JSON file named model_registration_details.json",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        help="Registered mlflow model id",
    )
    parser.add_argument(
        "--inference_payload",
        type=Path,
        help="Json file with inference endpoint payload.",
    )
    parser.add_argument(
        "--endpoint_name",
        type=str,
        help="Name of the endpoint",
    )
    parser.add_argument("--deployment_name", type=str, help="Name of the the deployment")
    parser.add_argument(
        "--instance_type",
        type=str,
        help="Compute instance type to deploy model",
        default="Standard_NC24s_v3",
    )
    parser.add_argument(
        "--instance_count",
        type=int,
        help="Number of compute instances to deploy model",
        default=1,
        choices=range(1, MAX_INSTANCE_COUNT),
    )
    parser.add_argument(
        "--max_concurrent_requests_per_instance",
        type=int,
        default=1,
        help="Maximum concurrent requests to be handled per instance",
    )
    parser.add_argument(
        "--request_timeout_ms",
        type=int,
        default=60000,  # 1min
        help="Request timeout in ms.",
    )
    parser.add_argument(
        "--max_queue_wait_ms",
        type=int,
        default=60000,  # 1min
        help="Maximum queue wait time of a request in ms",
    )
    parser.add_argument(
        "--failure_threshold_readiness_probe",
        type=int,
        default=10,
        help="No of times system will try after failing the readiness probe",
    )
    parser.add_argument(
        "--success_threshold_readiness_probe",
        type=int,
        default=1,
        help="The minimum consecutive successes for the readiness probe to be considered successful, after fail",
    )
    parser.add_argument(
        "--timeout_readiness_probe",
        type=int,
        default=10,
        help="The number of seconds after which the readiness probe times out",
    )
    parser.add_argument(
        "--period_readiness_probe",
        type=int,
        default=10,
        help="How often (in seconds) to perform the readiness probe",
    )
    parser.add_argument(
        "--initial_delay_readiness_probe",
        type=int,
        default=10,
        help="The number of seconds after the container has started before the readiness probe is initiated",
    )
    parser.add_argument(
        "--failure_threshold_liveness_probe",
        type=int,
        default=30,
        help="No of times system will try after failing the liveness probe",
    )
    parser.add_argument(
        "--timeout_liveness_probe",
        type=int,
        default=10,
        help="The number of seconds after which the liveness probe times out",
    )
    parser.add_argument(
        "--period_liveness_probe",
        type=int,
        default=10,
        help="How often (in seconds) to perform the liveness probe",
    )
    parser.add_argument(
        "--initial_delay_liveness_probe",
        type=int,
        default=10,
        help="The number of seconds after the container has started before the liveness probe is initiated",
    )
    parser.add_argument(
        "--egress_public_network_access",
        type=str,
        default="enabled",
        help="Secures the deployment by restricting interaction between deployment and Azure resources used by it",
    )
    parser.add_argument(
        "--model_deployment_details",
        type=str,
        help="Json file to which deployment details will be written",
    )
    # parse args
    args = parser.parse_args()
    logger.info(f"Args received {args}")
    print("args received ", args)

    # Validating passed input values
    if args.max_concurrent_requests_per_instance < 1:
        raise Exception("Arg max_concurrent_requests_per_instance cannot be less than 1")
    if args.request_timeout_ms < 1 or args.request_timeout_ms > MAX_REQUEST_TIMEOUT:
        raise Exception(f"Arg request_timeout_ms should lie between 1 and {MAX_REQUEST_TIMEOUT}")
    if args.max_queue_wait_ms < 1 or args.max_queue_wait_ms > MAX_REQUEST_TIMEOUT:
        raise Exception(f"Arg max_queue_wait_ms should lie between 1 and {MAX_REQUEST_TIMEOUT}")

    return args


def create_endpoint_and_deployment(ml_client, model_id, endpoint_name, deployment_name, args):
    """Create endpoint and deployment and return details."""
    endpoint = ManagedOnlineEndpoint(name=endpoint_name, auth_mode="key")

    # deployment
    deployment = ManagedOnlineDeployment(
        name=deployment_name,
        endpoint_name=endpoint_name,
        model=model_id,
        instance_type=args.instance_type,
        instance_count=args.instance_count,
        request_settings=OnlineRequestSettings(
            max_concurrent_requests_per_instance=args.max_concurrent_requests_per_instance,
            request_timeout_ms=args.request_timeout_ms,
            max_queue_wait_ms=args.max_queue_wait_ms,
        ),
        liveness_probe=ProbeSettings(
            failure_threshold=args.failure_threshold_liveness_probe,
            timeout=args.timeout_liveness_probe,
            period=args.period_liveness_probe,
            initial_delay=args.initial_delay_liveness_probe,
        ),
        readiness_probe=ProbeSettings(
            failure_threshold=args.failure_threshold_readiness_probe,
            success_threshold=args.success_threshold_readiness_probe,
            timeout=args.timeout_readiness_probe,
            period=args.period_readiness_probe,
            initial_delay=args.initial_delay_readiness_probe,
        ),
        egress_public_network_access=args.egress_public_network_access,
    )

    try:
        logger.info(f"Creating endpoint {endpoint_name}")
        ml_client.begin_create_or_update(endpoint).wait()
        endpoint = ml_client.online_endpoints.get(endpoint.name)
        logger.info(f"Endpoint created {endpoint.id}")
    except Exception as e:
        raise AzureMLException._with_error(
            AzureMLError.create(EndpointCreationError, exception=e)
        )

    try:
        logger.info(f"Creating deployment {deployment}")
        ml_client.online_deployments.begin_create_or_update(deployment).wait()
    except Exception as e:
        try:
            logger.error("Deployment failed. Printing deployment logs")
            logs = ml_client.online_deployments.get_logs(
                name=deployment_name,
                endpoint_name=endpoint_name,
                lines=MAX_DEPLOYMENT_LOG_TAIL_LINES
            )
            logger.error(logs)
        except Exception as ex:
            logger.error(f"Error in fetching deployment logs: {ex}")

        raise AzureMLException._with_error(
            AzureMLError.create(DeploymentCreationError, exception=e)
        )

    logger.info(f"Deployment successful. Updating endpoint to take 100% traffic for deployment {deployment_name}")

    # deployment to take 100% traffic
    endpoint.traffic = {deployment.name: 100}
    try:
        ml_client.begin_create_or_update(endpoint).wait()
        endpoint = ml_client.online_endpoints.get(endpoint.name)
    except Exception as e:
        error_msg = f"Error occured while updating endpoint traffic. Deployment should be usable. Exception - {e}"
        raise Exception(error_msg)

    logger.info(f"Endpoint updated to take 100% traffic for deployment {deployment_name}")
    return endpoint, deployment


@swallow_all_exceptions(logger)
def main():
    """Run main function."""
    args = parse_args()
    ml_client = get_mlclient()
    # get registered model id

    if args.model_id:
        model_id = str(args.model_id)
    elif args.registration_details_folder:
        registration_details_file = args.registration_details_folder/ComponentVariables.REGISTRATION_DETAILS_JSON_FILE
        if registration_details_file.exists():
            try:
                with open(registration_details_file) as f:
                    model_info = json.load(f)
                model_id = model_info["id"]
            except Exception as e:
                raise Exception(f"model_registration_details json file is missing model information {e}.")
        else:
            raise Exception(f"{ComponentVariables.REGISTRATION_DETAILS_JSON_FILE} is missing inside folder.")
    else:
        raise Exception("Arguments model_id and registration_details both are missing.")

    # Endpoint has following restrictions:
    # 1. Name must begin with lowercase letter
    # 2. Followed by lowercase letters, hyphen or numbers
    # 3. End with a lowercase letter or number

    # 1. Replace underscores and slashes by hyphens and convert them to lower case.
    # 2. Take 21 chars from model name and append '-' & timstamp(10chars) to it
    model_name = get_model_name(model_id)

    endpoint_name = re.sub("[^A-Za-z0-9]", "-", model_name).lower()[:21]
    endpoint_name = f"{endpoint_name}-{int(time.time())}"
    endpoint_name = endpoint_name

    endpoint_name = args.endpoint_name if args.endpoint_name else endpoint_name
    deployment_name = args.deployment_name if args.deployment_name else "default"

    endpoint, deployment = create_endpoint_and_deployment(
        ml_client=ml_client,
        endpoint_name=endpoint_name,
        deployment_name=deployment_name,
        model_id=model_id,
        args=args
    )

    if args.inference_payload:
        print("Invoking inference with test payload ...")
        try:
            response = ml_client.online_endpoints.invoke(
                endpoint_name=endpoint_name,
                deployment_name=deployment_name,
                request_file=args.inference_payload,
            )
            print(f"Response:\n{response}")
            logger.info(f"Endpoint invoked successfully with response :{response}")
        except Exception as e:
            raise AzureMLException._with_error(
                AzureMLError.create(OnlineEndpointInvocationError, exception=e)
            )

    print("Saving deployment details ...")

    # write deployment details to file
    endpoint_type = "aml_online_inference"
    deployment_details = {
        "endpoint_name": endpoint.name,
        "deployment_name": deployment.name,
        "endpoint_uri": endpoint.__dict__["_scoring_uri"],
        "endpoint_type": endpoint_type,
        "instance_type": args.instance_type,
        "instance_count": args.instance_count,
        "max_concurrent_requests_per_instance": args.max_concurrent_requests_per_instance,
    }
    json_object = json.dumps(deployment_details, indent=4)
    with open(args.model_deployment_details, "w") as outfile:
        outfile.write(json_object)
    logger.info("Saved deployment details in output json file.")


# run script
if __name__ == "__main__":
    # run main function
    main()

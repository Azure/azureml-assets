# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Run Model deployment module."""
import os
import argparse
import json
import logging
import re
import time
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    OnlineRequestSettings,
    ProbeSettings,
)
from azure.identity import ManagedIdentityCredential
from azure.ai.ml.identity import AzureMLOnBehalfOfCredential
from azureml.core import Run
from pathlib import Path


MAX_REQUEST_TIMEOUT = 90000
MAX_INSTANCE_COUNT = 20

# PROBE CONSTANTS
PROBE_FAILURE_THRESHOLD = 50
PROBE_SUCCESS_THRESHOLD = 50
PROBE_TIMEOUT = 500
PROBE_PERIOD = 500
PROBE_INITIAL_DELAY = 500


def parse_args():
    """Return arguments."""
    parser = argparse.ArgumentParser()

    # Defaults for managed online endpoint has been picked mostly from:
    # https://learn.microsoft.com/en-us/azure/machine-learning/reference-yaml-deployment-managed-online
    # Some of the defaults have been tweaked to cater to large models.

    # add arguments
    parser.add_argument(
        "--registration_details",
        type=Path,
        help="Json file that contains the ID of registered model to be deployed",
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
        choices=range(1, PROBE_FAILURE_THRESHOLD),
    )
    parser.add_argument(
        "--success_threshold_readiness_probe",
        type=int,
        default=1,
        help="The minimum consecutive successes for the readiness probe to be considered successful, after fail",
        choices=range(1, PROBE_SUCCESS_THRESHOLD),
    )
    parser.add_argument(
        "--timeout_readiness_probe",
        type=int,
        default=10,
        help="The number of seconds after which the readiness probe times out",
        choices=range(1, PROBE_TIMEOUT)
    )
    parser.add_argument(
        "--period_readiness_probe",
        type=int,
        default=10,
        help="How often (in seconds) to perform the readiness probe",
        choices=range(1, PROBE_PERIOD),
    )
    parser.add_argument(
        "--initial_delay_readiness_probe",
        type=int,
        default=10,
        help="The number of seconds after the container has started before the readiness probe is initiated",
        choices=range(1, PROBE_INITIAL_DELAY),
    )
    parser.add_argument(
        "--failure_threshold_liveness_probe",
        type=int,
        default=30,
        help="No of times system will try after failing the liveness probe",
        choices=range(1, PROBE_FAILURE_THRESHOLD),
    )
    parser.add_argument(
        "--timeout_liveness_probe",
        type=int,
        default=10,
        help="The number of seconds after which the liveness probe times out",
        choices=range(1, PROBE_TIMEOUT),
    )
    parser.add_argument(
        "--period_liveness_probe",
        type=int,
        default=10,
        help="How often (in seconds) to perform the liveness probe",
        choices=range(1, PROBE_PERIOD),
    )
    parser.add_argument(
        "--initial_delay_liveness_probe",
        type=int,
        default=10,
        help="The number of seconds after the container has started before the liveness probe is initiated",
        choices=range(1, PROBE_INITIAL_DELAY),
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
    print("args received ", args)

    # Validating passed input values
    if args.max_concurrent_requests_per_instance < 1:
        parser.error("Arg max_concurrent_requests_per_instance cannot be less than 1")
    if args.request_timeout_ms < 1 or args.request_timeout_ms > MAX_REQUEST_TIMEOUT:
        parser.error(f"Arg request_timeout_ms should lie between 1 and {MAX_REQUEST_TIMEOUT}")
    if args.max_queue_wait_ms < 1 or args.max_queue_wait_ms > MAX_REQUEST_TIMEOUT:
        parser.error(f"Arg max_queue_wait_ms should lie between 1 and {MAX_REQUEST_TIMEOUT}")

    return args


def get_ml_client():
    """Return ML client."""
    credential = AzureMLOnBehalfOfCredential()
    try:
        # Check if given credential can get token successfully.
        credential.get_token("https://management.azure.com/.default")
    except Exception as ex:
        # Fall back to ManagedIdentityCredential in case AzureMLOnBehalfOfCredential not work
        print(f"Failed to get obo credentials - {ex}")
        msi_client_id = os.environ.get("DEFAULT_IDENTITY_CLIENT_ID")
        credential = ManagedIdentityCredential(client_id=msi_client_id)
    try:
        credential.get_token("https://management.azure.com/.default")
    except Exception as ex:
        raise (f"Failed to get credentials : {ex}")
    run = Run.get_context(allow_offline=False)
    ws = run.experiment.workspace

    ml_client = MLClient(
        credential=credential,
        subscription_id=ws._subscription_id,
        resource_group_name=ws._resource_group,
        workspace_name=ws._workspace_name,
    )
    return ml_client


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
        print(f"Creating endpoint {endpoint_name}")
        ml_client.begin_create_or_update(endpoint).wait()
    except Exception as e:
        error_msg = f"Error occured while creating endpoint - {e}"
        logging.error(error_msg)
        raise Exception(error_msg)

    try:
        print(f"Creating deployment {deployment}")
        ml_client.online_deployments.begin_create_or_update(deployment).wait()
    except Exception as e:
        error_msg = f"Error occured while creating deployment - {e}"
        logging.error(error_msg)
        raise Exception(error_msg)

    print(f"Deployment successful. Updating endpoint to take 100% traffic for deployment {deployment_name}")

    # deployment to take 100% traffic
    endpoint.traffic = {deployment.name: 100}
    try:
        ml_client.begin_create_or_update(endpoint).wait()
        endpoint = ml_client.online_endpoints.get(endpoint.name)
    except Exception as e:
        error_msg = f"Error occured while updating endpoint traffic - {e}"
        logging.error(error_msg)
        raise Exception(error_msg)

    return endpoint, deployment


def main(args):
    """Run main function."""
    ml_client = get_ml_client()
    # get registered model id
    model_info = {}
    with open(args.registration_details) as f:
        model_info = json.load(f)
    model_id = model_info['id']
    model_name = model_info['name']

    # Endpoint has following restrictions:
    # 1. Name must begin with lowercase letter
    # 2. Followed by lowercase letters, hyphen or numbers
    # 3. End with a lowercase letter or number

    # 1. Replace underscores and slashes by hyphens and convert them to lower case.
    # 2. Take 21 chars from model name and append '-' & timstamp(10chars) to it
    endpoint_name = re.sub('[^A-Za-z0-9]', '-', model_name).lower()[:21]
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
        except Exception as e:
            print(f"Invocation failed with error: {e}")
            raise e

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


# run script
if __name__ == "__main__":
    args = parse_args()

    # run main function
    main(args)

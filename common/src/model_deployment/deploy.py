# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Run Model deployment module."""
import os
import argparse
from pathlib import Path
import json
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    OnlineRequestSettings,
)
from azure.identity import ManagedIdentityCredential
from azureml.core import Run

def parse_args():
    """Return arguments."""
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument(
        "--registration_details",
        type=str,
        help="Text file that contains the ID of registered model to be deployed",
    )
    parser.add_argument(
        "--endpoint_name",
        type=str,
        help="Name of the endpoint",
    )
    parser.add_argument(
        "--deployment_name", type=str, help="Name of the the deployment"
    )
    parser.add_argument(
        "--instance_type",
        type=str,
        help="Compute instance type to deploy model",
        default="Standard_F8s_v2",
    )
    parser.add_argument(
        "--instance_count",
        type=int,
        help="Number of compute instances to deploy model",
        default=1,
        choices=range(1, 20),
    )
    parser.add_argument(
        "--model_deployment_details",
        type=str,
        help="Json file to which deployment details will be written",
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
        default=5000,
        help="Request timeout in ms. Max limit is 90000.",
    )
    parser.add_argument(
        "--max_queue_wait_ms",
        type=int,
        default=500,
        help="Maximum queue wait time of a request in ms",
    )

    # parse args
    args = parser.parse_args()
    print("args received ", args)
    # return args

    # Validating passed input values
    if args.max_concurrent_requests_per_instance < 1:
        parser.error("Arg max_concurrent_requests_per_instance cannot be less than 1")
    if args.request_timeout_ms < 1:
        parser.error("Arg request_timeout_ms cannot be less than 1")
    if args.max_queue_wait_ms < 1:
        parser.error("Arg max_queue_wait_ms cannot be less than 1")
    return args

def get_endpoint(args):
    """Return online or batch endpoint."""
    endpoint = ManagedOnlineEndpoint(name=args.endpoint_name, auth_mode="key")
    print("Endpoint created with name ", endpoint.name)
    return endpoint

def get_ml_client():
    """Return ML client."""
    msi_client_id = os.environ.get("DEFAULT_IDENTITY_CLIENT_ID")
    credential = ManagedIdentityCredential(client_id=msi_client_id)

    run = Run.get_context(allow_offline=False)
    ws = run.experiment.workspace

    ml_client = MLClient(
        credential=credential,
        subscription_id=ws._subscription_id,
        resource_group_name=ws._resource_group,
        workspace_name=ws._workspace_name,
    )
    return ml_client

def main(args):
    """Run main function."""
    ml_client = get_ml_client()
    model_id = (Path(args.registration_details)).read_text()
    endpoint = get_endpoint(args)

    ml_client.begin_create_or_update(endpoint).wait()

    # deployment
    deployment = ManagedOnlineDeployment(
        name=args.deployment_name,
        endpoint_name=endpoint.name,
        model=model_id,
        instance_type=args.instance_type,
        instance_count=args.instance_count,
        request_settings=OnlineRequestSettings(
            max_concurrent_requests_per_instance=args.max_concurrent_requests_per_instance,
            request_timeout_ms=args.request_timeout_ms,
            max_queue_wait_ms=args.max_queue_wait_ms,
        ),
    )
    ml_client.online_deployments.begin_create_or_update(deployment).wait()

    print("Deployment done with name ", deployment.name)

    # deployment to take 100% traffic
    endpoint.traffic = {deployment.name: 100}
    ml_client.begin_create_or_update(endpoint).wait()

    # write deployment details to file
    endpoint_type = "aml_online_inference"
    endpoint_response = ml_client.online_endpoints.get(endpoint.name)
    deployment_details = {
        "endpoint_name": endpoint.name,
        "deployment_name": deployment.name,
        "endpoint_uri": endpoint_response.__dict__["_scoring_uri"],
        "endpoint_type": endpoint_type,
    }
    json_object = json.dumps(deployment_details, indent=4)
    with open(args.model_deployment_details, "w") as outfile:
        outfile.write(json_object)

# run script
if __name__ == "__main__":
    args = parse_args()

    # run main function
    main(args)

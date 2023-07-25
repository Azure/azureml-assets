# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Run Model deployment module."""
import os
import argparse
import json
from azure.ai.ml import MLClient
from azure.identity import ManagedIdentityCredential
from azure.ai.ml.identity import AzureMLOnBehalfOfCredential
from azureml.core import Run
from pathlib import Path


def parse_args():
    """Return arguments."""
    parser = argparse.ArgumentParser()

    # Defaults for managed online endpoint has been picked mostly from:
    # https://learn.microsoft.com/en-us/azure/machine-learning/reference-yaml-deployment-managed-online
    # Some of the defaults have been tweaked to cater to large models.

    # add arguments
    parser.add_argument(
        "--model_deployment_details",
        type=Path,
        help="Json file that contains the deployment details.",
    )
    parser.add_argument(
        "--endpoint_name",
        type=str,
        help="Name of the endpoint that needs to be deleted",
    )
    parser.add_argument(
        "--deployment_name",
        type=str,
        help="Name of the deployment that needs to be deleted",
    )
    parser.add_argument(
        "--endpoint_deletion_details",
        type=Path,
        help="Json file which contains information about the endpoint, deployment that get's deleted.",
    )
    args = parser.parse_args()
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


def main(args):
    """Run main function."""
    ml_client = get_ml_client()
    # get registered model id
    deployment_info = {}
    endpoint_name = ""
    if args.model_deployment_details:
        with open(args.model_deployment_details) as f:
            deployment_info = json.load(f)
        endpoint_name = deployment_info["endpoint_name"]
        ml_client.online_endpoints.begin_delete(name=endpoint_name).wait()
    else:
        if args.endpoint_name and not args.deployment_name:
            try:
                ml_client.online_endpoints.begin_delete(name=args.endpoint_name).wait()
            except Exception as e:
                print(f"Failed to delete endpoint : {endpoint_name}", e)
        elif args.endpoint_name and args.deployment_name:
            try:
                ml_client.online_deployments.begin_delete(name=args.deployment_name,
                                                          endpoint_name=args.endpoint_name).wait()
                deployments = list(ml_client.online_deployments.list(endpoint_name=args.endpoint_name))
                if not deployments:
                    ml_client.online_endpoints.begin_delete(name=args.endpoint_name).wait()
            except Exception as e:
                print(e)


# run script
if __name__ == "__main__":
    args = parse_args()

    # run main function
    main(args)

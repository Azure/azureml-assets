# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Clone a pipeline and substitute all components with the ones from specified environment.

Typical usage:
    python clone_pipeline.py -p PIPELINE_ID -e ENVIRONMENT_ID [-n NEW_PIPELINE_NAME]

    i.e.
    python clone_pipeline.py -p 12345678-1234-1234-1234-123456789012 -e private-embedding-env -n new-pipeline-name

"""

import argparse
import shutil
import tempfile
import webbrowser
from pathlib import Path

from azure.ai.ml import MLClient
from azure.ai.ml.entities import BuildContext, Environment
from azure.identity import (
    AzureCliCredential,
    ChainedTokenCredential,
    DeviceCodeCredential,
)

# Default values
SUBSCRIPTION_ID = "f375b912-331c-4fc5-8e9f-2d7205e3e036"
RESOURCE_GROUP = "rag-release-validation-rg"
WORKSPACE_NAME = "rag-release-validation-ws"

CUSTOM_ENVIRONMENT_NAME = "custom-llm-embedding-env"


def main(
    subscription_id,
    resource_group,
    workspace_name,
    pipeline_id,
    environment_name,
    new_pipeline_name,
) -> None:
    """Clone a pipeline and substitute all components with the ones from specified environment."""
    credential = ChainedTokenCredential(AzureCliCredential(), DeviceCodeCredential())
    ml_client = MLClient(
        credential=credential,
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace_name,
    )

    # get the pipeline
    from azure.ai.ml._restclient.v2023_04_01_preview.models import ListViewType

    list_of_jobs = ml_client.jobs.list(
        parent_job_name=pipeline_id, list_view_type=ListViewType.ARCHIVED_ONLY
    )

    for job in list_of_jobs:
        print(f"{job.display_name}({job.id})")
        component = job._to_component()
        print(component._type)

    #


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-s",
        "--subscription-id",
        type=str,
        required=False,
        default=SUBSCRIPTION_ID,
    )
    parser.add_argument(
        "-r",
        "--resource-group",
        type=str,
        required=False,
        default=RESOURCE_GROUP,
    )
    parser.add_argument(
        "-w",
        "--workspace-name",
        type=str,
        required=False,
        default=WORKSPACE_NAME,
    )
    parser.add_argument(
        "-p",
        "--pipeline-id",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-e",
        "--environment-name",
        type=str,
        required=False,
        default=CUSTOM_ENVIRONMENT_NAME,
    )
    parser.add_argument(
        "-n",
        "--new-pipeline-name",
        type=str,
        required=False,
    )

    try:
        args = parser.parse_args()
    except argparse.ArgumentError as e:
        print(e)
        parser.print_help()
        exit(1)

    if args.pipeline_id is None:
        print("Please provide run id of the pipeline.")
        parser.print_help()
        exit(1)

    main(
        args.subscription_id,
        args.resource_group,
        args.workspace_name,
        args.pipeline_id,
        args.environment_name,
        args.new_pipeline_name,
    )

    print("Done.")

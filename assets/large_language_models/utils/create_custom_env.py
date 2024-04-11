# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Create customer environments with given Dockerfile and/or wheel packages.

Two use scenarios:
1. Using your own Dockerfile
python create_custom_env.py [-s SUBSCRIPTION_ID] [-r RESOURCE_GROUP] [-w WORKSPACE_NAME] [-d DOCKER_DIR] [-n CUSTOM_ENVIRONMENT_NAME]

2. Adding wheel packages under package directory. BTW. build wheel file "python setup.py bdist_wheel"
python create_custom_env.py [-s SUBSCRIPTION_ID] [-r RESOURCE_GROUP] [-w WORKSPACE_NAME] [-p PACKAGE_DIR] [-n CUSTOM_ENVIRONMENT_NAME]
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

PROD_REGISTRY_NAME = "azureml"
PROD_ENVIRONMENT_NAME = "llm-rag-embeddings"


def main(
    subscription_id: str,
    resource_group: str,
    workspace_name: str,
    docker_dir: Path,
    package_dir: Path,
    environment_name: str,
) -> None:
    credential = ChainedTokenCredential(AzureCliCredential(), DeviceCodeCredential())

    ml_client = MLClient(
        credential=credential,
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace_name,
    )

    # Create temp directory
    with tempfile.TemporaryDirectory() as temp_dir:
        staging_dir = Path(temp_dir)
        print(f"Temp directory: {staging_dir}")

        # copy Dockerfile to temp directory
        if docker_dir is not None:
            shutil.copy(docker_dir, staging_dir)

        if not (staging_dir / "Dockerfile").exists():
            # Create Dockerfile if not provided
            with open(staging_dir / "Dockerfile", "w") as f:
                f.write(
                    f"FROM mcr.microsoft.com/{PROD_REGISTRY_NAME}/curated/{PROD_ENVIRONMENT_NAME}@latest AS base\n"
                )

        # Find all .whl files and copy them into staging_dir / "wheels" directory
        wheel_dir = staging_dir / "wheels"
        if package_dir is not None:
            for wheel in package_dir.glob("*.whl"):
                shutil.copy(wheel, wheel_dir)

        # Adding wheel packages to Dockerfile
        if package_dir is not None:
            with open(f"{staging_dir}/Dockerfile", "a") as f:
                for wheel in wheel_dir.glob("*.whl"):
                    f.write(f"COPY ./wheels/{wheel.name} /wheels/\n")
                f.write("\nRUN pip install --force-reinstall /wheels/*.whl\n")

        env = Environment(
            name=environment_name,
            description="Custom environment.",
            build=BuildContext(path=staging_dir, dockerfile_path="Dockerfile"),
            version="1",
        )

        # get latest environment version by name
        latest_env = ml_client.environments.get(environment_name, label="latest")
        if latest_env is not None:
            env.version = f"{int(latest_env.version) + 1}"

        result = ml_client.environments.create_or_update(env)
        wsid = ml_client.workspaces.get(workspace_name).id
        url = f"https://ml.azure.com/environments/{result.name}/version/{result.version}?wsid={wsid}"

        print()
        print(f"Creating custom environment '{env.name}':{env.version}...")
        print(url)
        webbrowser.open_new_tab(url)

        # clean up temp directory
    print("Done.")


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
        "-d",
        "--docker-dir",
        type=str,
        required=False,
        help="Path to the directory, where the Dockerfile is located.",
    )
    parser.add_argument(
        "-p",
        "--package-dir",
        type=str,
        required=False,
        help="Path to the directory, where the customer .whl files are located.",
    )
    parser.add_argument(
        "-n",
        "--environment-name",
        type=str,
        required=False,
        default=CUSTOM_ENVIRONMENT_NAME,
    )

    try:
        args = parser.parse_args()
    except argparse.ArgumentError as e:
        print(e)
        parser.print_help()
        exit(1)

    if args.docker_dir is None and args.package_dir is None:
        print("Neither docker directory nor package directory is provided.")
        print("Please provide at least one of them.")
        parser.print_help()
        exit(1)

    main(
        args.subscription_id,
        args.resource_group,
        args.workspace_name,
        args.staging_dir,
        args.environment_name,
    )

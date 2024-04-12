# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Create RAG components pointing to a custom environment.

Usage:
python create_rag_components.py [-e ENVIRONMENT_STR] [-v COMPONENT_VERSION] \
    [-s SUBSCRIPTION_ID] [-r RESOURCE_GROUP] [-w WORKSPACE_NAME]

Note:
CUSTOM_ENVIRONMENT: The name and version of the custom environment to use.
    format: <environment_name>:<environment_version>
    Default: custom-llm-embedding-env:latest
COMPONENT_VERSION: Latest version increased by 0.0.1 if not specified.
"""

import argparse
import shutil
import tempfile
from pathlib import Path
from typing import List, Union

import yaml
from azure.ai.ml import MLClient, load_component
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

# Local git repo
GIT_REPO_ROOT = Path(f"{__file__}").resolve().parent.parent.parent.parent


def main(
    subscription_id: str,
    resource_group: str,
    workspace_name: str,
    environment_string: str,
    component_version: str = None,
) -> None:
    """Main function to create RAG components pointing to a custom environment."""
    credential = ChainedTokenCredential(AzureCliCredential(), DeviceCodeCredential())

    ml_client = MLClient(
        credential=credential,
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace_name,
    )

    # Get a custom environment
    environment_name, environment_version = environment_string.split(":")
    if environment_version == "latest" or environment_version is None:
        environment = ml_client.environments.get(environment_name, label="latest")
    else:
        environment = ml_client.environments.get(
            environment_name, version=environment_version
        )

    if environment is None:
        raise ValueError(f"Environment {environment_string} not found.")

    print(f"Using environment {environment.name}:{environment.version}.")

    # Create temp directory
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Temp directory: {temp_dir}")

        # copy component definition to the temp folder
        src = GIT_REPO_ROOT / "assets" / "large_language_models" / "rag" / "components"
        src.absolute()
        component_root = Path(temp_dir) / "components"
        shutil.copytree(src, component_root, dirs_exist_ok=True)

        # Create a list of component name and version
        component_versions = {}
        for component_spec in component_root.glob("**/spec.yaml"):
            # Open the yaml file in read mode
            with open(component_spec, "r") as f:
                component = yaml.safe_load(f)
                current_component_name = component["name"]
                if component_version is None:
                    current_component_version = "0.0.0"

                    try:
                        current_component = ml_client.components.get(
                            current_component_name, label="latest"
                        )
                    except Exception:
                        current_component = None
                        print(f"Component {current_component_name} not found.")

                    if current_component is not None:
                        current_component_version = current_component.version
                        version_segments = current_component_version.split(".")
                        major_version_str = ".".join(version_segments[:-1])
                        minor_version_str = f"{int(version_segments[-1]) + 1}"
                        incremented_version = f"{major_version_str}.{minor_version_str}"
                        component_versions[current_component_name] = incremented_version
                    else:
                        component_versions[current_component_name] = "0.0.1"
                else:
                    component_versions[current_component_name] = component_version

        # Update the component version and environment settings in the component spec
        for component_spec in component_root.glob("**/spec.yaml"):
            # Read
            with open(component_spec, "r") as f:
                data = yaml.safe_load(f)
            # Update
            data["version"] = component_versions[data["name"]]
            data["environment"] = f"azureml:{environment.name}:{environment.version}"
            # Write
            with open(component_spec, "w") as f:
                yaml.safe_dump(data, f)

        # Create or update the components
        for component_spec in component_root.glob("**/spec.yaml"):
            try:
                component = ml_client.components.create_or_update(load_component(component_spec))
            except Exception as e:
                print(f"Failed to create component from {component_spec}. Error: {e}")
                continue
            print(f"Component {component.name}:{component.version} created successfully .")

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
        "-e",
        "--environment-string",
        type=str,
        required=True,
        help="Path to the directory, where the Dockerfile is located.",
    )
    parser.add_argument(
        "-v",
        "--component-version",
        type=str,
        required=False,
        help="Latest version increased by 0.0.1 if not specified.",
    )

    try:
        args = parser.parse_args()
    except argparse.ArgumentError as e:
        print(e)
        parser.print_help()
        exit(1)

    main(
        subscription_id=args.subscription_id,
        resource_group=args.resource_group,
        workspace_name=args.workspace_name,
        environment_string=args.environment_string,
        component_version=args.component_version,
    )

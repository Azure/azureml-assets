# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Script to upgrade aml-benchmark components before release.

This script updates all the required parts in a component and finally
prints the regular expression to be used in the release.
"""

import os
import re
import argparse
from typing import List, Union, Dict, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm
from yaml import safe_load
from azure.ai.ml.constants._common import AzureMLResourceType
from azure.ai.ml.constants._component import NodeType
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.core.exceptions import ResourceNotFoundError

AML_BENCH_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
REG_ML_CLIENT = MLClient(credential=DefaultAzureCredential(), registry_name="azureml")
AVOID_COMPONENT_FOLDERS = {"batch-resource-manager"}


def _get_all_components_spec() -> List[str]:
    """Get all components spec."""
    components = []
    for root, dirs, _ in os.walk(os.path.join(AML_BENCH_DIR, "components")):
        for folder in dirs:
            if folder in AVOID_COMPONENT_FOLDERS:
                print(f"Skipping {folder}.")
                continue
            spec_path = os.path.join(root, folder, "spec.yaml")
            if os.path.exists(spec_path):
                components.append(spec_path)
    return components


def _get_asset_latest_version(
    asset_name: str,
    asset_type: Union[AzureMLResourceType.COMPONENT, AzureMLResourceType.ENVIRONMENT],
    asset_version: str,
) -> str:
    """Get component latest version."""
    try:
        if asset_type == AzureMLResourceType.COMPONENT:
            asset = REG_ML_CLIENT.components.get(name=asset_name, label="latest")
        elif asset_type == AzureMLResourceType.ENVIRONMENT:
            asset = REG_ML_CLIENT.environments.get(name=asset_name, label="latest")
    except ResourceNotFoundError:
        return asset_version
    return asset.version


def _get_bumped_version(version: str) -> str:
    """Return bumped version."""
    version_arr = list(map(int, version.split(".")))
    version_arr[-1] += 1
    return ".".join(map(str, version_arr))


def __replace_pipeline_comp_job_version(match: re.Match) -> str:
    """Replace version for job in pipeline component."""
    component_name_with_registry = match.group(1)
    component_name = component_name_with_registry.split(":")[-1]
    latest_version = _get_asset_latest_version(
        asset_name=component_name,
        asset_type=AzureMLResourceType.COMPONENT,
        asset_version=match.group(2),
    )
    new_version = _get_bumped_version(latest_version)
    return f"component: {component_name_with_registry}:{new_version}"


def _upgrade_component_env(spec: Dict[str, Any], spec_str: str, env_version: str) -> str:
    """Upgrade component's environment."""
    type = spec["type"]

    if type == NodeType.COMMAND or type == NodeType.PARALLEL:
        if type == NodeType.COMMAND:
            env_arr = spec["environment"].split("/")
        elif type == NodeType.PARALLEL:
            env_arr = spec["task"]["environment"].split("/")
        else:
            raise ValueError(f"Invalid type {type}.")

        # if env_version is '', keep the version as is, no replacement needed
        if env_version == "":
            return spec_str
        # else if env_version is latest, get the latest version
        elif env_version == "latest":
            latest_version = _get_asset_latest_version(
                asset_name=env_arr[-3],
                asset_type=AzureMLResourceType.ENVIRONMENT,
                asset_version=env_arr[-1],
            )
        # else, use the provided version
        else:
            latest_version = env_version

        if env_arr[-1] == "latest":
            env_arr[-2] = "versions"
        env_arr[-1] = latest_version
        spec_str = re.sub(
            pattern=r"environment: .*",
            repl=f"environment: {'/'.join(env_arr)}",
            string=spec_str,
        )

    elif type == NodeType.PIPELINE:
        spec_str = re.sub(
            pattern=r"component: (\S+:\S+):(\d+\.\d+\.\d+)",
            repl=__replace_pipeline_comp_job_version,
            string=spec_str,
        )

    return spec_str


def _upgrade_component(
    component_path: str,
    env_version: str,
) -> Tuple[bool, Union[str, None], str, Optional[str]]:
    """Upgrade component spec.

    :param component_path: Path to component spec.
    :param env_version: Environment version to upgrade to. Defaults to 'latest', which will update env to the latest
        version. Use '' to keep the version as is. Use a specific version to upgrade to that version."
    :return: Tuple of (error, error_message, component_path, component_name).
    """
    is_error = False
    error = None
    name = None
    try:
        with open(component_path, "r") as file:
            spec = safe_load(file)
            file.seek(0)
            spec_str = file.read()
        name = spec["name"]

        # bump component version
        latest_version = _get_asset_latest_version(
            asset_name=name,
            asset_type=AzureMLResourceType.COMPONENT,
            asset_version=spec["version"],
        )
        new_version = _get_bumped_version(latest_version)
        spec["version"] = new_version
        spec_str = re.sub(
            pattern=r"version: .*", repl=f"version: {new_version}", string=spec_str
        )

        # bump component's environment only where version is hardcoded
        spec_str = _upgrade_component_env(spec, spec_str, env_version)

        with open(component_path, "w") as file:
            file.write(spec_str)
    except Exception as e:
        is_error = True
        error = str(e)
    return is_error, error, component_path, name


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=f"{__file__}")
    parser.add_argument(
        "--env_version",
        type=str,
        default="latest",
        help=(
            "Environment version to upgrade to. Defaults to latest, which will update env to the latest version."
            "Use '' to keep the version as is. Use a specific version to upgrade to that version."
        ),
    )
    args, _ = parser.parse_known_args()
    return args


def main(env_version: str = "latest") -> None:
    """
    Entry function.

    :param env_version: Environment version to upgrade to. Defaults to 'latest', which will update env to the latest
        version. Use '' to keep the version as is. Use a specific version to upgrade to that version."
    :return: None
    """
    components = _get_all_components_spec()

    # upgrade components concurrently with tqdm
    max_allowed_threads = max(1, os.cpu_count() - 1)
    print(
        f"\nUpgrading components with {max_allowed_threads} threads... \nPlease wait and check for errors."
    )
    with ThreadPoolExecutor(max_workers=max_allowed_threads) as executor:
        results = list(
            tqdm(
                executor.map(
                    lambda args: _upgrade_component(*args),
                    [(component, env_version) for component in components]
                ),
                total=len(components),
            )
        )

    # check for errors
    error_count = 0
    error_mssgs = []
    regex = "component/("
    for is_error, error_mssg, _, comp_name in results:
        if is_error:
            error_count += 1
            mssg = (
                f"#{error_count}. Error in upgrading component '{comp_name}'. "
                f"Error details: \n\n{error_mssg}"
            )
            error_mssgs.append(mssg)
        else:
            regex += f"{comp_name}|"
    # remove the last "|" and add the end of the regex
    regex = regex[:-1] + ")/.+"

    # print errors
    if error_count > 0:
        print(f"\U0001F61E Errors found {error_count}:")
        print(
            "------------------------------------ERRORS------------------------------------"
        )
        print("\n".join(error_mssgs))
        print(
            "\n\nPlease fix the errors and re-run the script to get the regular expression."
        )
    else:
        print("\U0001F603 No errors found!")
        print(f"\n\nRegular Expression: {regex}")


if __name__ == "__main__":
    args = _parse_args()
    main(env_version=args.env_version)

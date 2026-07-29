# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Script to upgrade components before release.

This script updates all the required parts in a component and finally
prints the regular expression to be used in the release. Components
are read from components.yaml file.
"""

import os
import re
from typing import List, Union, Dict, Any, Tuple, Optional, Set
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time

from tqdm import tqdm
from yaml import safe_load
from azure.ai.ml.constants._common import AzureMLResourceType
from azure.ai.ml.constants._component import NodeType
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.core.exceptions import ResourceNotFoundError


ASSETS_DIR = Path(__file__).resolve().parents[3]
REG_ML_CLIENT = MLClient(credential=DefaultAzureCredential(), registry_name="azureml")
FIRST_VERSION = "0.0.1"
CACHE: Dict[str, str] = {}

_components_yaml_path = Path(__file__).resolve().parents[0] / "components.yaml"
with open(_components_yaml_path, "r", encoding="utf-8") as file:
    OWNED_COMPONENT_NAMES: Set[str] = set(safe_load(file)["component"])


def _get_components_spec_path() -> List[str]:
    """Get all components spec path that requires update."""
    # get required components' spec paths
    component_paths = []
    for root, _, files in os.walk(ASSETS_DIR):
        if "spec.yaml" in files:
            asset_path = os.path.join(root, "spec.yaml")
            with open(asset_path, "r", encoding="utf-8") as file:
                spec = safe_load(file)
            if spec.get("name", None) not in OWNED_COMPONENT_NAMES:
                continue
            component_paths.append(asset_path)
    return component_paths


def _get_bumped_version(version: str, increment: bool = True) -> str:
    """
    Return bumped version.

    :param version: Version to bump.
    :param increment: If True, increment the last part of the version. Else, decrement the last part of the version.
    :return: Bumped version.
    """
    version_arr = list(map(int, version.split(".")))
    if increment:
        version_arr[-1] += 1
    else:
        version_arr[-1] -= 1
    return ".".join(map(str, version_arr))


def _get_asset_latest_version(
    asset_name: str,
    asset_type: Union[AzureMLResourceType.COMPONENT, AzureMLResourceType.ENVIRONMENT],
) -> Optional[str]:
    """Get component latest version."""
    global CACHE
    if asset_name in CACHE:
        return str(CACHE[asset_name])
    try:
        if asset_type == AzureMLResourceType.COMPONENT:
            asset = REG_ML_CLIENT.components.get(name=asset_name, label="latest")
        elif asset_type == AzureMLResourceType.ENVIRONMENT:
            asset = REG_ML_CLIENT.environments.get(name=asset_name, label="latest")
    except ResourceNotFoundError:
        return None
    CACHE[asset_name] = asset.version
    return asset.version


def __replace_pipeline_comp_job_version(match: re.Match) -> str:
    """Replace version for job in pipeline component."""
    component_name_with_registry = match.group(1)
    _component_name = component_name_with_registry.split(":")[-1]
    latest_version = _get_asset_latest_version(
        asset_name=_component_name,
        asset_type=AzureMLResourceType.COMPONENT,
    )
    if latest_version is None:
        new_version = match.group(2)
        new_version = new_version if new_version is not None else FIRST_VERSION
    else:
        if _component_name in OWNED_COMPONENT_NAMES:
            new_version = _get_bumped_version(latest_version)
        else:
            new_version = latest_version
    return f"component: {component_name_with_registry}:{new_version}"


def _upgrade_component_env(spec: Dict[str, Any], spec_str: str) -> str:
    """Upgrade component's environment."""
    type = spec["type"]

    if type == NodeType.COMMAND or type == NodeType.PARALLEL:
        if type == NodeType.COMMAND:
            env_arr = spec["environment"].split("/")
        elif type == NodeType.PARALLEL:
            env_arr = spec["task"]["environment"].split("/")
        else:
            raise ValueError(f"Invalid type {type}.")

        latest_version = _get_asset_latest_version(
            asset_name=env_arr[-3],
            asset_type=AzureMLResourceType.ENVIRONMENT,
        )
        if latest_version is None:
            latest_version = env_arr[-1]

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
            pattern=r"component: ([^:@\s]+:[^:@\s]+)(?::(\d+\.\d+\.\d+)|@latest)?",
            repl=__replace_pipeline_comp_job_version,
            string=spec_str,
        )

    return spec_str


def _upgrade_component(
    component_path: str,
) -> Tuple[bool, Union[str, None], str, Optional[str]]:
    """Upgrade component spec.

    :param component_path: Path to component spec.
    :return: Tuple of (error, error_message, component_path, component_name).
    """
    is_error = False
    error = None
    name = None
    try:
        with open(component_path, "r", encoding="utf-8") as file:
            spec = safe_load(file)
            file.seek(0)
            spec_str = file.read()
        name = spec["name"]

        # bump component version
        latest_version = _get_asset_latest_version(
            asset_name=name,
            asset_type=AzureMLResourceType.COMPONENT,
        )
        if latest_version is None:
            new_version = FIRST_VERSION
        else:
            new_version = _get_bumped_version(latest_version)
        spec["version"] = new_version
        spec_str = re.sub(
            pattern=r"version: .*", repl=f"version: {new_version}", string=spec_str
        )

        # bump component's environment only where version is hardcoded
        spec_str = _upgrade_component_env(spec, spec_str)

        with open(component_path, "w", encoding="utf-8") as file:
            file.write(spec_str)
    except Exception as e:
        is_error = True
        error = str(e)
    return is_error, error, component_path, name


def main() -> None:
    """Entry function."""
    component_spec_paths = _get_components_spec_path()

    max_allowed_threads = 1
    print(
        f"\nUpgrading {len(component_spec_paths)} components with {max_allowed_threads} thread(s)... "
        "\nPlease wait and check for errors."
    )

    start_time = time.time()
    with ThreadPoolExecutor(max_workers=max_allowed_threads) as executor:
        results = list(
            tqdm(
                executor.map(_upgrade_component, component_spec_paths),
                total=len(component_spec_paths),
            )
        )
    end_time = time.time()

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
        print(
            "\U0001F603 No errors found! Took {:.2f} seconds.".format(
                end_time - start_time
            )
        )
        print(f"\n\nRegular Expression: {regex}")


if __name__ == "__main__":
    main()

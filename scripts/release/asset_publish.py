# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Python script to publish assets."""

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path
from string import Template
from subprocess import PIPE, STDOUT, run
from tempfile import TemporaryDirectory
from collections import defaultdict
from typing import Dict, List, Union
import azureml.assets as assets
from azureml.assets.model.mlflow_utils import MLFlowModelUtils
import azureml.assets.util as util
import yaml
from azureml.assets.config import PathType
from azureml.assets.model import ModelDownloadUtils
from azureml.assets.util import logger
from azure.ai.ml import MLClient, load_component, load_model
from azure.ai.ml.entities import Component, Environment, Model


ASSET_ID_TEMPLATE = Template("azureml://registries/$registry_name/$asset_type/$asset_name/versions/$version")
TEST_YML = "tests.yml"
PUBLISH_ORDER = [assets.AssetType.ENVIRONMENT, assets.AssetType.COMPONENT, assets.AssetType.MODEL]
WORKSPACE_ASSET_PATTERN = re.compile(r"^(?:azureml:)?(.+)(?::(.+)|@(.+))$")
REGISTRY_ENV_PATTERN = re.compile(r"^azureml://registries/.+/environments/(.+)/(?:versions/(.+)|labels/(.+))")


def find_test_files(dir: Path):
    """Find test files in the directory."""
    test_jobs = []

    for test in dir.iterdir():
        logger.print("processing test folder: " + test.name)
        with open(test / TEST_YML) as fp:
            data = yaml.load(fp, Loader=yaml.FullLoader)
            for test_group in data.values():
                for test_job in test_group['jobs'].values():
                    if 'job' in test_job:
                        test_jobs.append((test / test_job['job']).as_posix())
    return test_jobs


def preprocess_test_files(test_jobs, asset_ids: dict):
    """Preprocess test files to generate asset ids."""
    for test_job in test_jobs:
        logger.print(f"processing test job: {test_job}")
        with open(test_job) as fp:
            data = yaml.load(fp, Loader=yaml.FullLoader)
            for job_name, job in data['jobs'].items():
                asset_name = job['component']
                logger.print(f"processing asset {asset_name}")
                if asset_name in asset_ids:
                    job['component'] = asset_ids.get(asset_name)
                    logger.print(
                        f"for job {job_name}, the new asset id: {job['component']}")
            with open(test_job, "w") as file:
                yaml.dump(data, file, default_flow_style=False,
                          sort_keys=False)


def update_spec(asset: Union[Component, Environment, Model], spec_path: Path) -> bool:
    """Update the yaml spec file with updated properties in asset.

    :param asset: Asset loaded using load_*(component, environemnt, model) method.
    :type asset: Union[Component, Environment, Model]
    :param spec_path: path to asset spec file
    :type spec_path: Path
    :return: True if spec was successfully updated
    :rtype: bool
    """
    try:
        asset_dict = json.loads(json.dumps(asset._to_dict()))
        util.dump_yaml(asset_dict, spec_path)
        return True
    except Exception as e:
        logger.log_error(f"Failed to update spec => {e}")
    return False


def prepare_model(model_config: assets.ModelConfig, spec_file_path: Path, model_dir: Path) -> bool:
    """Prepare Model.

    :param model_config: Model Config object
    :type model_config: assets.ModelConfig
    :param spec_file_path: path to model spec file
    :type spec_file_path: Path
    :param model_dir: path of directory where model is present locally or can be downloaded to.
    :type model_dir: Path
    :return: If model can be published to registry.
    :rtype: bool
    """
    try:
        model = load_model(spec_file_path)
        # TODO: temp fix before restructuring what attributes are required in model config and spec.
        model.type = model_config.type.value
    except Exception as e:
        logger.error(f"Error in loading model spec file at {spec_file_path} => {e}")
        return False

    if model_config.path.type == PathType.LOCAL:
        model.path = os.path.abspath(Path(model_config.path.uri).resolve())
        return update_spec(model, spec_file_path)

    if model_config.type == assets.ModelType.CUSTOM:
        can_publish_model = ModelDownloadUtils.download_model(model_config.path.type, model_config.path.uri, model_dir)
        if can_publish_model:
            model.path = model_dir
            can_publish_model = update_spec(model, spec_file_path)

    elif model_config.type == assets.ModelType.MLFLOW:
        can_publish_model = ModelDownloadUtils.download_model(model_config.path.type, model_config.path.uri, model_dir)
        if can_publish_model:
            model.path = model_dir / MLFlowModelUtils.MLFLOW_MODEL_PATH
            if not model_config.flavors:
                # try fetching flavors from MLModel file
                mlmodel_file_path = model.path / MLFlowModelUtils.MLMODEL_FILE_NAME
                try:
                    mlmodel = util.load_yaml(file_path=mlmodel_file_path)
                    model.flavors = mlmodel.get("flavors")
                except Exception as e:
                    logger.log_error(f"Error loading flavors from MLmodel file at: {mlmodel_file_path} => {e}")
            can_publish_model = update_spec(model, spec_file_path)

    else:
        logger.print(model_config.type.value, assets.ModelType.MLFLOW)
        can_publish_model = False
        logger.log_error(f"Model type {model_config.type} not supported")

    return can_publish_model


def validate_update_command_component(
    component: Component,
    spec_path: Path,
    final_version: str,
    registry_name: str,
) -> bool:
    """Validate and update command component spec.

    :param component: A command component
    :type component: Component
    :param spec_path: Path of loaded component
    :type spec_path: Path
    :param final_version: Final version string used to register component
    :type final_version: str
    :param registry_name: name of the registry to publish component to
    :type registry_name: str
    :return: True for successful validation and update
    :rtype: bool
    """
    env = component.environment
    match = None
    for pattern in [REGISTRY_ENV_PATTERN, WORKSPACE_ASSET_PATTERN]:
        if (match := pattern.match(env)) is not None:
            break

    if not match:
        logger.print(f"Env ID doesn't match workspace or registry pattern in {asset.spec_with_path}")
        return False

    # both ws and registry env follows the same grouping
    env_name, env_version, env_label = match.group(1), match.group(2), match.group(3)
    logger.print(f"Env name: {env_name}, version: {env_version}, label: {env_label}")

    if env_label:
        # TODO: Add fetching env from label
        # https://github.com/Azure/azureml-assets/issues/415
        logger.print("Unexpected !!! Registering a component with env label is not supported.")
        return False

    env = None
    # Check if component's env is registered
    registered_envs = get_registered_asset_versions(assets.AssetType.ENVIRONMENT, env_name, registry_name)    
    env = next((x for x in registered_envs if x['version'] in [env_version, final_version] ), None)

    if not env:
        logger.print(f"Could not find a registered env for {component.name}. Please retry again!!!")
        return False

    component.environment = env["id"]
    if not update_spec(component, spec_path):
        logger.print(f"Component update failed for asset spec path: {asset.spec_path}")
        return False


def run_command(cmd: List[str]):
    """Run the command for and return result."""
    result = run(cmd, stdout=PIPE, stderr=PIPE, encoding=sys.stdout.encoding, errors="ignore")
    return result


def asset_list_command(
    asset_type: str,
    asset_name: str,
    registry_name: str,
) -> List[str]:
    """Command to list of registered asset versions."""
    cmd = [
        "az", "ml", asset_type, "list",
        "--name", asset_name,
        "--registry-name", registry_name,
    ]
    print(cmd)
    return cmd


def asset_publish_command(
    asset_type: str,
    asset_path: str,
    registry_name: str,
    version: str,
    resource_group: str,
    workspace: str,
    debug_mode: bool = None,
) -> List[str]:
    """Assemble the az cli command."""
    cmd = [
        shutil.which("az"), "ml", asset_type, "create",
        "--file", asset_path,
        "--registry-name", registry_name,
        "--version", version,
    ]
    if resource_group:
        cmd.extend(["--resource-group", resource_group])
    if workspace:
        cmd.extend(["--workspace", workspace])
    if debug_mode:
        cmd.append("--debug")
    print(cmd)
    return cmd


def publish_asset(
    asset,
    registry_name,
    resource_group,
    workspace_name,
    version,
    failure_list,
    debug_mode: bool = None
):
    """ Publish asset to registry."""
    registered_assets = get_registered_asset_versions(asset.type.value, asset.name, registry_name, return_dict=True)
    if version in registered_assets:
        print(f"Version already registered. Skipping publish for asset: {asset.name}")
        return

    cmd = asset_publish_command(
        asset.type.value, str(asset.spec_with_path),
        registry_name, version, resource_group, workspace_name, debug_mode
    )

    # Run command
    result = run_command(cmd, failure_list, debug_mode)
    if debug_mode:
        # Capture and redact output
        redacted_output = re.sub(r"Bearer.*", "", result.stdout)
        print(redacted_output)

    if result.returncode != 0:
        print(f"Error creating {asset.type.value} : {asset.name}")
        failure_list.append(asset)


def get_registered_asset_versions(asset_type: str, asset_name: str, registry_name: str, return_dict=False) -> Union[Dict, List]:
    """Return list/dict of registered asset versions."""
    result = run_command(asset_list_command(
        asset_type=asset_type,
        asset_name=asset_name,
        registry_name=registry_name,
    ))
    if result.returncode != 0:
        print(f"Error in listing asset version. stdout:\n{result.stdout}")
        result = "[]"
    registered_assets = json.loads(result.stdout)
    if return_dict:
        return {x['version']:x for x in registered_assets}
    return registered_assets


def _str2bool(v: str) -> bool:
    """
    Parse boolean-ish values.

    See https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--registry-name", required=True,
                        type=str, help="the registry name")
    parser.add_argument("-s", "--subscription-id",
                        type=str, help="the subscription ID")
    parser.add_argument("-g", "--resource-group", type=str,
                        help="the resource group name")
    parser.add_argument("-w", "--workspace", type=str,
                        help="the workspace name")
    parser.add_argument("-a", "--assets-directory",
                        required=True, type=Path, help="the assets directory")
    parser.add_argument("-t", "--tests-directory",
                        type=Path, help="the tests directory")
    parser.add_argument("-v", "--version-suffix", type=str,
                        help="the version suffix")
    parser.add_argument("-l", "--publish-list", type=Path,
                        help="the path of the publish list file")
    parser.add_argument("-f", "--failed-list", type=Path,
                        help="the path of the failed assets list file")
    parser.add_argument(
        "-d", "--debug", type=_str2bool, nargs="?",
        const=True, default=False, help="debug mode",
    )
    args = parser.parse_args()

    registry_name = args.registry_name
    subscription_id = args.subscription_id
    resource_group = args.resource_group
    workspace = args.workspace
    tests_dir = args.tests_directory
    assets_dir = args.assets_directory
    passed_version = args.version_suffix
    publish_list_file = args.publish_list
    failed_list_file = args.failed_list
    debug_mode = args.debug
    asset_ids = {}
    logger.print("publishing assets")

    # Load publishing list from deploy config
    if publish_list_file:
        with open(publish_list_file) as fp:
            config = yaml.load(fp, Loader=yaml.FullLoader)
            publish_list = config.get('create', {})
    else:
        publish_list = {}

    # Check publishing list
    if not publish_list:
        logger.log_warning("The create list is empty.")
        exit(0)
    logger.print(f"create list: {publish_list}")

    failure_list = []
    all_assets = util.find_assets(input_dirs=assets_dir)
    assets_by_type = defaultdict(list)
    for asset in all_assets:
        assets_by_type[asset.type.value].append(asset)

    for publish_asset_type in PUBLISH_ORDER:
        logger.print(f"now publishing {publish_asset_type.value}s.")
        if publish_asset_type.value not in publish_list:
            continue
        for asset in assets_by_type.get(publish_asset_type.value, []):
            asset_names = publish_list.get(asset.type.value, [])
            if not ("*" in asset_names or asset.name in asset_names):
                logger.print(
                    f"Skipping registering asset {asset.name} because it is not in the publish list")
                continue
            logger.print(f"Registering {asset}")
            final_version = asset.version + "-" + \
                passed_version if passed_version else asset.version
            logger.print(f"final version: {final_version}")
            asset_ids[asset.name] = ASSET_ID_TEMPLATE.substitute(
                registry_name=registry_name,
                asset_type=f"{asset.type.value}s",
                asset_name=asset.name,
                version=final_version,
            )

            # Handle specific asset types
            if asset.type == assets.AssetType.COMPONENT:
                # load component and check if environment exists
                logger.print(f"spec's path: {asset.spec_with_path}")
                component = load_component(asset.spec_with_path)
                if component.type == "command":
                    if not validate_update_command_component(
                        component, asset.spec_with_path, final_version, registry_name
                    ):
                        failure_list.append(asset)
                        continue
            elif asset.type == assets.AssetType.MODEL:
                try:
                    model_config = asset.extra_config_as_object()
                    with TemporaryDirectory() as tempdir:
                        if not prepare_model(model_config, asset.spec_with_path, Path(tempdir)):
                            raise Exception(f"Could not prepare model at {asset.spec_with_path}")
                except Exception as e:
                    logger.log_error(f"Model prepare exception. Error => {e}")
                    failure_list.append(asset)
                    continue

            # publish asset
            publish_asset(
                asset=asset,
                version=final_version,
                registry_name=registry_name,
                resource_group=resource_group,
                workspace_name=workspace,
                failure_list=failure_list,
                debug_mode=debug_mode
            )

    if len(failure_list) > 0:
        failed_assets = defaultdict(list)
        for asset in failure_list:
            failed_assets[asset.type.value].append(asset.name)

        for asset_type, asset_names in failed_assets.items():
            logger.log_warning(f"Failed to register {asset_type}s: {asset_names}")
        # the following dump process will generate a yaml file for the report
        # process in the end of the publishing script
        with open(failed_list_file, "w") as file:
            yaml.dump(failed_assets, file, default_flow_style=False, sort_keys=False)

    if tests_dir:
        logger.print("locating test files")
        test_jobs = find_test_files(tests_dir)

        logger.print("preprocessing test files")
        preprocess_test_files(test_jobs, asset_ids)
        logger.print("finished preprocessing test files")
    else:
        logger.log_warning("Test files not found")

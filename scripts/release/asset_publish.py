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
from subprocess import run
from tempfile import TemporaryDirectory
from collections import defaultdict
from typing import Dict, List, Tuple, Union
import azureml.assets as assets
from azureml.assets.model.mlflow_utils import MLFlowModelUtils
import azureml.assets.util as util
import yaml
from azureml.assets.config import AssetConfig, PathType
from azureml.assets.model import ModelDownloadUtils
from azureml.assets.util import logger
from azure.ai.ml import load_component, load_model
from azure.ai.ml.entities import Component, Environment, Model


ASSET_ID_TEMPLATE = Template("azureml://registries/$registry_name/$asset_type/$asset_name/versions/$version")
TEST_YML = "tests.yml"
PROD_SYSTEM_REGISTRY = "azureml"
CREATE_ORDER = [assets.AssetType.ENVIRONMENT, assets.AssetType.COMPONENT, assets.AssetType.MODEL]
WORKSPACE_ASSET_PATTERN = re.compile(r"^(?:azureml:)?(.+)(?::(.+)|@(.+))$")
REGISTRY_ENV_PATTERN = re.compile(r"^azureml://registries/(.+)/environments/(.+)/(?:versions/(.+)|labels/(.+))")
REGISTRY_ASSET_TEMPLATE = Template("^azureml://registries/(.+)/${asset_type}s/(.+)/(?:versions/(.+)|labels/(.+))")
BEARER = r"Bearer.*"


def find_test_files(dir: Path):
    """Find test files in the directory."""
    test_jobs = []

    for test in dir.iterdir():
        logger.print(f"Processing test folder {test.name}")
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
        logger.print(f"Processing test job {test_job}")
        with open(test_job) as fp:
            data = yaml.load(fp, Loader=yaml.FullLoader)
            for job_name, job in data['jobs'].items():
                asset_name = job['component']
                logger.print(f"Processing asset {asset_name}")
                if asset_name in asset_ids:
                    job['component'] = asset_ids.get(asset_name)
                    logger.print(f"For job {job_name}, the new asset id is {job['component']}")
            with open(test_job, "w") as file:
                yaml.dump(data, file, default_flow_style=False,
                          sort_keys=False)


def sanitize_output(input: str) -> str:
    """Return sanitized string."""
    # Remove sensitive token
    sanitized_output = re.sub(BEARER, "", input)
    return sanitized_output


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
        logger.log_error(f"Failed to update spec: {e}")
    return False


def prepare_model(model_config: assets.ModelConfig, spec_file_path: Path, model_dir: Path) -> bool:
    """Prepare model.

    :param model_config: Model Config object
    :type model_config: assets.ModelConfig
    :param spec_file_path: path to model spec file
    :type spec_file_path: Path
    :param model_dir: path of directory where model is present locally or can be downloaded to.
    :type model_dir: Path
    :return: Model successfully prepared for creation in registry.
    :rtype: bool
    """
    try:
        model = load_model(spec_file_path)
        model.type = model_config.type.value
    except Exception as e:
        logger.error(f"Error in loading model spec file at {spec_file_path}: {e}")
        return False

    model_description_file_path = Path(spec_file_path).parent / model_config.description
    logger.print(f"model_description_file_path {model_description_file_path}")
    if os.path.exists(model_description_file_path):
        with open(model_description_file_path) as f:
            model_description = f.read()
            model.description = model_description
    else:
        logger.print("description file does not exist")

    if model_config.path.type == PathType.LOCAL:
        model.path = os.path.abspath(Path(model_config.path.uri).resolve())
        return update_spec(model, spec_file_path)

    if model_config.type == assets.ModelType.CUSTOM:
        success = ModelDownloadUtils.download_model(model_config.path.type, model_config.path.uri, model_dir)
        if success:
            model.path = model_dir
            success = update_spec(model, spec_file_path)
    elif model_config.type == assets.ModelType.MLFLOW:
        success = ModelDownloadUtils.download_model(model_config.path.type, model_config.path.uri, model_dir)
        if success:
            model.path = model_dir / MLFlowModelUtils.MLFLOW_MODEL_PATH
            if not model.flavors:
                # try fetching flavors from MLModel file
                mlmodel_file_path = model.path / MLFlowModelUtils.MLMODEL_FILE_NAME
                try:
                    mlmodel = util.load_yaml(file_path=mlmodel_file_path)
                    model.flavors = mlmodel.get("flavors")
                except Exception as e:
                    logger.log_error(f"Error loading flavors from MLmodel file at {mlmodel_file_path}: {e}")
            success = update_spec(model, spec_file_path)
    else:
        logger.log_error(f"Model type {model_config.type.value} not supported")
        success = False

    return success


def validate_and_prepare_pipeline_component(
    spec_path: Path,
    final_version: str,
    registry_name: str,
) -> bool:
    """Validate and update pipeline component spec.

    :param spec_path: Path of loaded component
    :type spec_path: Path
    :param final_version: Final version string used to create component
    :type final_version: str
    :param registry_name: name of the registry to create component in
    :type registry_name: str
    :return: True for successful validation and update
    :rtype: bool
    """
    with open(spec_path) as f:
        pipeline_dict = yaml.safe_load(f)

    jobs = pipeline_dict['jobs']
    logger.print(f"Preparing pipeline component {pipeline_dict['name']}")
    updated_jobs = {}

    for job_name, job_details in jobs.items():
        logger.print(f"job {job_name}")
        try:
            name, version, label, registry = get_parsed_details_from_asset_uri(
                assets.AssetType.COMPONENT.value, job_details['component'])
        except Exception as e:
            logger.log_error(e)
            return False

        logger.print(
            "component details:\n"
            + f"name: {name}\n"
            + f"version: {version}\n"
            + f"label: {label}\n"
            + f"registry: {registry}"
        )

        if registry and registry not in [PROD_SYSTEM_REGISTRY, registry_name]:
            logger.log_error(
                "Registry name for component's URI must be either "
                + f"'{registry_name}' or '{PROD_SYSTEM_REGISTRY}'. Got '{registry}'"
            )
            return False

        # Check if component's env exists
        registry_name = registry or registry_name
        asset_details = None
        for ver in [version, final_version]:
            if (asset_details := get_asset_details(
                assets.AssetType.COMPONENT.value, name, ver, registry_name
            )) is not None:
                break

        if not asset_details:
            logger.log_warning(
                f"dependent component {name} with version {version} not found in registry {registry}"
            )
            return False

        # logger.print(asset_details)
        updated_jobs[job_name] = job_details
        updated_jobs[job_name]['component'] = asset_details["id"]

    pipeline_dict['jobs'] = updated_jobs

    try:
        util.dump_yaml(pipeline_dict, spec_path)
    except Exception:
        logger.log_error(f"Component update failed for asset spec path: {asset.spec_path}")
        return False
    return True


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
    :param final_version: Final version string used to create component
    :type final_version: str
    :param registry_name: name of the registry to create component in
    :type registry_name: str
    :return: True for successful validation and update
    :rtype: bool
    """
    try:
        env_name, env_version, env_label, env_registry_name = get_parsed_details_from_asset_uri(
            assets.AssetType.ENVIRONMENT.value, component.environment)
    except Exception as e:
        logger.log_error(e)
        return False

    logger.print(
        f"Env name: {env_name}, version: {env_version}, label: {env_label}, env_registry_name: {env_registry_name}"
    )

    if env_registry_name and env_registry_name not in [PROD_SYSTEM_REGISTRY, registry_name]:
        logger.log_error(
            "Registry name for component's env URI must be either "
            + f"'{registry_name}' or '{PROD_SYSTEM_REGISTRY}'. Got '{env_registry_name}'"
        )
        return False

    registry_name = env_registry_name or registry_name

    if env_label:
        # TODO: Add fetching env from label
        # https://github.com/Azure/azureml-assets/issues/415
        logger.log_error("Creating a component with env label is not supported")
        return False

    env = None
    # Check if component's env exists
    for version in [env_version, final_version]:
        if (env := get_asset_details(
            assets.AssetType.ENVIRONMENT.value, env_name, version, registry_name
        )) is not None:
            break

    if not env:
        logger.log_error(f"Could not find the env for {component.name}")
        return False

    env_id = env["id"]
    logger.print(f"Updating component env to {env_id}")
    component.environment = env_id
    if not update_spec(component, spec_path):
        logger.log_error(f"Component update failed for asset spec path: {asset.spec_path}")
        return False
    return True


def run_command(cmd: List[str]):
    """Run the command for and return result."""
    result = run(cmd, capture_output=True, encoding=sys.stdout.encoding, errors="ignore")
    return result


def asset_create_command(
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
    return cmd


def create_asset(
    asset,
    registry_name,
    resource_group,
    workspace_name,
    version,
    failure_list,
    debug_mode: bool = None
):
    """Create asset in registry."""
    cmd = asset_create_command(
        asset.type.value, str(asset.spec_with_path),
        registry_name, version, resource_group, workspace_name, debug_mode
    )

    # Run command
    result = run_command(cmd)
    if debug_mode:
        # Capture and redact output
        logger.print(f"Executed: {cmd}")
        redacted_output = sanitize_output(result.stdout)
        if redacted_output:
            logger.print(f"STDOUT: {redacted_output}")

    if result.returncode != 0:
        redacted_err = sanitize_output(result.stderr)
        logger.log_error(f"Error creating {asset.type.value} {asset.name}: {redacted_err}")
        failure_list.append(asset)


def get_asset_details(
    asset_type: str,
    asset_name: str,
    asset_version: str,
    registry_name: str,
) -> Dict:
    """Get asset details."""
    cmd = [
        "az", "ml", asset_type, "show",
        "--name", asset_name,
        "--version", asset_version,
        "--registry-name", registry_name,
    ]
    result = run_command(cmd)
    if result.returncode != 0:
        logger.log_error(f"Failed to get asset details: {result.stderr}")
        return None
    return json.loads(result.stdout)


def get_parsed_details_from_asset_uri(asset_type: str, asset_uri: str) -> Tuple[str, str, str, str]:
    """Validate asset URI and return parsed details. Exception is raised for an invalid URI.

    :param asset_type: Valid values are component, environment and model
    :type asset_type: str
    :param asset_uri: A workspace or registry asset URI to parse
    :type asset_uri: str
    :return:
        A tuple with asset `name`, `version`, `label`, and `registry_name` in order.
        `label` and `registry_name` will be None for workspace URI.
    :rtype: Tuple
    """
    REGISTRY_ASSET_PATTERN = re.compile(REGISTRY_ASSET_TEMPLATE.substitute(asset_type=asset_type))
    asset_registry_name = None
    if (match := REGISTRY_ASSET_PATTERN.match(asset_uri)) is not None:
        asset_registry_name, asset_name, asset_version, asset_label = match.groups()
    elif (match := WORKSPACE_ASSET_PATTERN.match(asset_uri)) is not None:
        asset_name, asset_version, asset_label = match.groups()
    else:
        raise Exception(f"{asset_uri} doesn't match workspace or registry pattern.")
    return asset_name, asset_version, asset_label, asset_registry_name


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

    # Load publishing list from deploy config
    if publish_list_file:
        with open(publish_list_file) as fp:
            config = yaml.load(fp, Loader=yaml.FullLoader)
            create_list = config.get('create', {})
    else:
        create_list = {}

    # Check create list
    if not create_list:
        logger.log_warning("The create list is empty.")
        exit(0)
    logger.print(f"create list: {create_list}")

    failure_list = []
    all_assets = util.find_assets(input_dirs=assets_dir)
    assets_by_type: Dict[str, List[AssetConfig]] = defaultdict(list)
    for asset in all_assets:
        assets_by_type[asset.type.value].append(asset)

    for create_asset_type in CREATE_ORDER:
        logger.print(f"Creating {create_asset_type.value}s.")
        if create_asset_type.value not in create_list:
            continue

        assets_to_publish = assets_by_type.get(create_asset_type.value, [])
        if create_asset_type == assets.AssetType.COMPONENT:
            # sort component list to keep pipline components at the end in publishing list
            # this is a temporary solution as a pipeline component can have another pipeline component as dependency
            logger.print("updating components publishing order")
            assets_to_publish.sort(key=lambda x: x.spec_as_object().type == assets.ComponentType.PIPELINE.value)

        for asset in assets_to_publish:
            with TemporaryDirectory() as work_dir:
                asset_names = create_list.get(asset.type.value, [])
                if not ("*" in asset_names or asset.name in asset_names):
                    logger.print(
                        f"Skipping asset {asset.name} because it is not in the create list")
                    continue
                final_version = asset.version + "-" + \
                    passed_version if passed_version else asset.version
                logger.print(f"Creating {asset.name} {final_version}")
                asset_ids[asset.name] = ASSET_ID_TEMPLATE.substitute(
                    registry_name=registry_name,
                    asset_type=f"{asset.type.value}s",
                    asset_name=asset.name,
                    version=final_version,
                )

                # Handle specific asset types
                if asset.type == assets.AssetType.COMPONENT:
                    # load component and check if environment exists
                    component = load_component(asset.spec_with_path)
                    if component.type == assets.ComponentType.PIPELINE.value:
                        if not validate_and_prepare_pipeline_component(
                            asset.spec_with_path, final_version, registry_name
                        ):
                            failure_list.append(asset)
                            continue
                    elif component.type == assets.ComponentType.COMMAND.value:
                        if not validate_update_command_component(
                            component, asset.spec_with_path, final_version, registry_name
                        ):
                            failure_list.append(asset)
                            continue
                elif asset.type == assets.AssetType.MODEL:
                    # Check if model already exists
                    final_version = asset.version
                    if get_asset_details(asset.type.value, asset.name, final_version, registry_name):
                        logger.print(f"{asset.name} {final_version} already exists, skipping")
                        continue
                    try:
                        model_config = asset.extra_config_as_object()
                        if not prepare_model(model_config, asset.spec_with_path, Path(work_dir)):
                            raise Exception(f"Could not prepare model at {asset.spec_with_path}")
                    except Exception as e:
                        logger.log_error(f"Model prepare exception: {e}")
                        failure_list.append(asset)
                        continue

                # Create asset
                create_asset(
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
            logger.log_warning(f"Failed to create {asset_type}s: {asset_names}")
        # the following dump process will generate a yaml file for the report
        # process in the end of the publishing script
        with open(failed_list_file, "w") as file:
            yaml.dump(failed_assets, file, default_flow_style=False, sort_keys=False)

    if tests_dir:
        logger.print("Locating test files")
        test_jobs = find_test_files(tests_dir)

        logger.print("Preprocessing test files")
        preprocess_test_files(test_jobs, asset_ids)
        logger.print("Finished preprocessing test files")
    else:
        logger.log_warning("Test files not found")

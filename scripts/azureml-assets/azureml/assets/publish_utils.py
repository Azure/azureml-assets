# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Utility functions used to publish assets."""

import json
import re
import shutil
import sys
import azureml.assets.util as util
from pathlib import Path
from string import Template
from subprocess import CompletedProcess, run
from tempfile import TemporaryDirectory
from typing import Dict, List, Tuple, Union
from azureml.assets.config import AssetConfig, AssetType, ComponentType, ModelConfig, DataConfig
from azureml.assets.deployment_config import AssetVersionUpdate
from azureml.assets.model.registry_utils import CopyUpdater, prepare_model, update_metadata, \
    prepare_data, RegistryUtils
from azureml.assets.util import logger
from azureml.assets.util.util import resolve_from_file_for_asset
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Component, Environment, Model
from ruamel.yaml import YAML


PROD_SYSTEM_REGISTRY = "azureml"
WORKSPACE_ASSET_PATTERN = re.compile(r"^(?:azureml:)?(.+)(?::(.+)|@(.+))$")
REGISTRY_ENV_PATTERN = re.compile(r"^azureml://registries/(.+)/environments/(.+)/(?:versions/(.+)|labels/(.+))")
REGISTRY_ASSET_TEMPLATE = Template("^azureml://registries/(.+)/$asset_type/(.+)/(?:versions/(.+)|labels/(.+))")
BEARER = r"Bearer.*"
LATEST_LABEL = "latest"


def sanitize_output(input: str) -> str:
    """Return sanitized string."""
    # Remove sensitive token
    sanitized_output = re.sub(BEARER, "", input)
    return sanitized_output


def update_spec(asset: Union[Component, Environment, Model], spec_path: Path) -> bool:
    """Update the yaml spec file with updated properties in asset.

    :param asset: Asset loaded using load_*(component, environment, model) method.
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


def prepare_model_for_registration(
    model_config: ModelConfig,
    spec_file_path: Path,
    temp_dir: Path,
    ml_client: MLClient,
    copy_updater: CopyUpdater = None,
    output_level: str = "essential",
) -> bool:
    """Prepare model.

    :param model_config: Model Config object
    :type model_config: ModelConfig
    :param spec_file_path: path to model spec file
    :type spec_file_path: Path
    :param temp_dir: temp dir for model operation
    :type temp_dir: Path
    :param ml_client: MLClient object
    :type ml_client: MLClient
    :param copy_updater: CopyUpdater object to update files during azcopy
    :type copy_updater: CopyUpdater
    :param output_level: Parameter for azcopy output verbosity level
    :type output_level: str
    :return: Model successfully prepared for creation in registry.
    :rtype: bool
    """
    model, success = prepare_model(
        spec_path=spec_file_path, model_config=model_config, temp_dir=temp_dir, ml_client=ml_client,
        copy_updater=copy_updater, output_level=output_level
    )
    if success:
        success = update_spec(model, spec_file_path)
        logger.print(f"updated spec file? {success}")
    return success


def prepare_data_for_registration(
    data_config: DataConfig,
    spec_file_path: Path,
    temp_dir: Path,
    ml_client: MLClient,
    copy_updater: CopyUpdater = None,
    output_level: str = "essential",
) -> bool:
    """Prepare data.

    :param data_config: Data Config object
    :type data_config: DataConfig
    :param spec_file_path: path to data spec file
    :type spec_file_path: Path
    :param temp_dir: temp dir for data operation
    :type temp_dir: Path
    :param ml_client: MLClient object
    :type ml_client: MLClient
    :param copy_updater: CopyUpdater object to update files during azcopy
    :type copy_updater: CopyUpdater
    :param output_level: Parameter for azcopy output verbosity level
    :type output_level: str
    :return: Data successfully prepared for creation in registry.
    :rtype: bool
    """
    data, success = prepare_data(
        spec_path=spec_file_path, data_config=data_config, temp_dir=temp_dir, ml_client=ml_client,
        copy_updater=copy_updater, output_level=output_level
    )
    if success:
        success = update_spec(data, spec_file_path)
        logger.print(f"updated spec file? {success}")
    return success


def validate_and_prepare_pipeline_component(
    spec_path: Path,
    registry_name: str,
    version_template: str = None
) -> bool:
    """Validate and update pipeline component spec.

    :param spec_path: Path of loaded component
    :type spec_path: Path
    :param registry_name: name of the registry to create component in
    :type registry_name: str
    :param version_template: version template
    :type version_suffix: str
    :return: True for successful validation and update
    :rtype: bool
    """
    with open(spec_path) as f:
        try:
            pipeline_dict = YAML().load(f)
        except Exception:
            logger.log_error(f"Error in loading component spec at {spec_path}")
            return False

    jobs = pipeline_dict['jobs']
    logger.print(f"Preparing pipeline component {pipeline_dict['name']}")
    updated_jobs = {}

    for job_name, job_details in jobs.items():
        logger.print(f"job {job_name}")
        if not job_details.get('component'):
            # if-else or inline component
            logger.print(f"component not defined for job {job_name}")
            updated_jobs[job_name] = job_details
            continue

        try:
            name, version, label, registry = get_parsed_details_from_asset_uri(
                AssetType.COMPONENT.value, job_details['component'])
        except Exception as e:
            logger.log_error(e)
            return False

        logger.print(
            "Parsed component asset URI details:\n"
            + f"name: {name}\n"
            + f"version: {version}\n"
            + f"label: {label}\n"
            + f"registry: {registry}"
        )

        if registry and registry not in [PROD_SYSTEM_REGISTRY, registry_name]:
            logger.log_warning(
                f"Dependencies should exist in '{registry_name}' or '{PROD_SYSTEM_REGISTRY}'. "
                f"The URI for component '{name}' references registry '{registry}', "
                "and publishing will fail if the release process does not have read access to it."
            )

        # If workspace asset URI is used, use registry we're creating the component in
        if not registry:
            logger.print(f"Workspace asset URI was used, using component from registry {registry}")
            registry = registry_name

        if not version:
            logger.log_error(
                f"Component {name} parsed label {label} from the asset URI. Labels are not supported "
                f"for job components. Please specify a valid version instead."
            )
            return False

        # Check if component's env exists
        final_version = util.apply_version_template(version, version_template)
        asset_details = None
        for ver in [version, final_version]:
            if (asset_details := get_asset_details(
                AssetType.COMPONENT.value, name, ver, registry
            )) is not None:
                break

        if not asset_details:
            logger.log_warning(
                f"dependent component {name} with version {version} not found in registry {registry}"
            )
            return False

        updated_jobs[job_name] = job_details
        updated_jobs[job_name]['component'] = asset_details["id"]

    pipeline_dict['jobs'] = updated_jobs

    try:
        util.dump_yaml(pipeline_dict, spec_path)
    except Exception:
        logger.log_error(f"Component update failed for asset spec path: {spec_path}")
        return False
    return True


def get_environment_asset_id(
    environment_id: str,
    registry_name: str,
    version_template: str = None
) -> Union[object, None]:
    """Convert an environment reference into a full asset ID.

    :param environment_id: Environment asset ID, in short or long form
    :type environment_id: str
    :param registry_name: Name of the registry to create component in
    :type registry_name: str
    :param version_template: Version template
    :type version_template: str
    :return: Environment's full asset ID if successful, else None
    :rtype: Union[str, None]
    """
    try:
        env_name, env_version, env_label, env_registry_name = get_parsed_details_from_asset_uri(
            AssetType.ENVIRONMENT.value, environment_id)
    except Exception as e:
        logger.log_error(e)
        return False

    logger.print(
        f"Env name: {env_name}, version: {env_version}, label: {env_label}, env_registry_name: {env_registry_name}"
    )

    if env_registry_name and env_registry_name not in [PROD_SYSTEM_REGISTRY, registry_name]:
        logger.log_warning(
            f"Dependencies should exist in '{registry_name}' or '{PROD_SYSTEM_REGISTRY}'. "
            f"The URI for environment '{env_name}' references registry '{env_registry_name}', "
            "and publishing will fail if the release process does not have read access to it."
        )

    registry_name = env_registry_name or registry_name

    if env_label:
        if env_label == LATEST_LABEL:
            # TODO: Use a more direct approach like this, when supported by Azure CLI:
            # az ml environment show --name sklearn-1.1-ubuntu20.04-py38-cpu --registry-name azureml --label latest
            versions = get_asset_versions(AssetType.ENVIRONMENT.value, env_name, registry_name)
            if versions:
                # List is returned with the latest version at the beginning
                env_version = versions[0]
            else:
                logger.log_error(f"Unable to retrieve versions for env {env_name}")
                return False
        else:
            # TODO: Add fetching env from other labels
            # https://github.com/Azure/azureml-assets/issues/415
            logger.log_error(f"Creating a component with env label {env_label} is not supported")
            return False

    env = None
    # Get environment
    versions_to_try = [env_version]
    if version_template:
        versions_to_try.append(util.apply_version_template(env_version, version_template))
    for version in versions_to_try:
        if (env := get_asset_details(
            AssetType.ENVIRONMENT.value, env_name, version, registry_name
        )) is not None:
            return env['id']

    logger.log_error(f"Environment {env_name} not found in {registry_name}; tried version(s) {versions_to_try}")
    return None


def validate_update_component(
    spec_path: Path,
    registry_name: str,
    version_template: str = None
) -> bool:
    """Validate and update component spec.

    :param spec_path: Path of loaded component
    :type spec_path: Path
    :param registry_name: name of the registry to create component in
    :type registry_name: str
    :param version_template: version template
    :type version_template: str
    :return: True for successful validation and update
    :rtype: bool
    """
    with open(spec_path) as f:
        try:
            component_dict = YAML().load(f)
        except Exception:
            logger.log_error(f"Error in loading component spec at {spec_path}")
            return False

    component_name = component_dict['name']
    logger.print(f"Preparing component {component_name}")

    # Handle command and parallel components
    if 'environment' in component_dict:
        # Command component
        obj_with_env = component_dict
    elif 'task' in component_dict and 'environment' in component_dict['task']:
        # Parallel component
        obj_with_env = component_dict['task']
    else:
        logger.log_error(f"Environment reference not found in {component_name}")
        return False

    # Update environment reference
    current_env_id = obj_with_env['environment']
    new_env_id = get_environment_asset_id(current_env_id, registry_name, version_template)
    if new_env_id is not None:
        if current_env_id != new_env_id:
            logger.print(f"Updating environment to {new_env_id}")
            obj_with_env['environment'] = new_env_id
        else:
            logger.print(f"Existing environment reference {current_env_id} is valid")
    else:
        return False

    # Update spec file
    try:
        util.dump_yaml(component_dict, spec_path)
    except Exception:
        logger.log_error(f"Component update failed for asset spec path: {spec_path}")
        return False
    return True


def run_command(cmd: List[str]) -> CompletedProcess:
    """Run the command for and return result."""
    result = run(cmd, capture_output=True, encoding=sys.stdout.encoding, errors="ignore")
    return result


def asset_create_command(
    asset_type: str,
    asset_path: str,
    registry_name: str,
    version: str,
    debug: bool = None,
) -> List[str]:
    """Assemble the az cli command."""
    cmd = [
        shutil.which("az"), "ml", asset_type, "create",
        "--file", asset_path,
        "--registry-name", registry_name,
        "--version", version,
    ]
    if debug:
        cmd.append("--debug")
    return cmd


def create_asset_cli(
    asset: AssetConfig,
    registry_name: str,
    version: str,
    debug: bool = None
) -> bool:
    """Create asset in registry."""
    cmd = asset_create_command(
        asset.type.value, str(asset.spec_with_path),
        registry_name, version, debug
    )

    # Run command
    result = run_command(cmd)
    if debug:
        # Capture and redact output
        logger.print(f"Executed: {cmd}")
        redacted_output = sanitize_output(result.stdout)
        if redacted_output:
            logger.print(f"STDOUT: {redacted_output}")

    if result.returncode != 0:
        redacted_err = sanitize_output(result.stderr)
        logger.log_error(f"Error creating {asset.type.value} {asset.name}: {redacted_err}")
        return False
    return True


def get_asset_versions(
    asset_type: str,
    asset_name: str,
    registry_name: str,
) -> List[str]:
    """Get asset versions from registry."""
    cmd = [
        "az", "ml", asset_type, "list",
        "--name", asset_name,
        "--registry-name", registry_name,
    ]
    result = run_command(cmd)
    if result.returncode != 0:
        logger.log_error(f"Failed to list assets: {result.stderr}")
        return []
    return [a['version'] for a in json.loads(result.stdout)]


def get_asset_details(
    asset_type: str,
    asset_name: str,
    asset_version: str,
    registry_name: str,
) -> Dict:
    """Get asset details."""
    logger.print(f"Getting asset details for {asset_type} {asset_name} {asset_version} in {registry_name}")
    cmd = [
        "az", "ml", asset_type, "show",
        "--name", asset_name,
        "--version", asset_version,
        "--registry-name", registry_name,
    ]
    result = run_command(cmd)
    if result.returncode != 0:
        if "Could not find asset" not in result.stderr:
            # Don't show the error if it's expected for new assets
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
    REGISTRY_ASSET_PATTERN = re.compile(REGISTRY_ASSET_TEMPLATE.substitute(
                                        asset_type=RegistryUtils.pluralize_asset_type(asset_type)))
    asset_registry_name = None
    if (match := REGISTRY_ASSET_PATTERN.match(asset_uri)) is not None:
        asset_registry_name, asset_name, asset_version, asset_label = match.groups()
    elif (match := WORKSPACE_ASSET_PATTERN.match(asset_uri)) is not None:
        asset_name, asset_version, asset_label = match.groups()
    else:
        raise Exception(f"{asset_uri} doesn't match workspace or registry pattern.")
    return asset_name, asset_version, asset_label, asset_registry_name


def stringify_dictionary(dictionary: Dict):
    """Convert the type of values to string."""
    new_dict = {}
    for name, value in dictionary.items():
        new_dict[name] = json.dumps(value) if isinstance(value, dict) else str(value)
    return new_dict


def update_asset_metadata(asset: AssetConfig, ml_client: MLClient, allow_no_op_update: bool = False):
    """Update the mutable metadata of asset."""
    if asset.type in [AssetType.COMPONENT, AssetType.DATA, AssetType.MODEL]:
        asset_name = asset.name
        asset_version = asset.version
        spec_path = asset.spec_with_path

        tags_to_update = None
        try:
            with open(spec_path) as f:
                asset_spec = YAML().load(f)
                tags = asset_spec.get("tags", {})
                properties = asset_spec.get("properties", {})

                if asset.type == AssetType.MODEL:
                    model_config = asset.extra_config_as_object()
                    tags = {k: resolve_from_file_for_asset(model_config, v) for k, v in tags.items()}
                    description = model_config.description

                # convert tags, properties value to string
                tags = stringify_dictionary(tags)
                properties = stringify_dictionary(properties)
                tags_to_update = {"replace": tags}
                properties_to_update = {"add": properties}

                if asset.type in [AssetType.COMPONENT, AssetType.DATA]:
                    description = asset_spec.get("description", None)
        except Exception as e:
            logger.log_error(f"Failed to get tags for {asset.type.value} {asset_name}: {e}")

        update_metadata(
            name=asset_name,
            version=asset_version,
            update=AssetVersionUpdate(
                versions=[asset_version],
                tags=tags_to_update,
                properties=properties_to_update,
                description=description
            ),
            ml_client=ml_client,
            asset_type=asset.type,
            allow_no_op_update=allow_no_op_update,
        )
    else:
        logger.print(f"Skipping metadata update of {asset.name}. Not supported for type {asset.type}")


def create_asset(asset: AssetConfig, registry_name: str, ml_client: MLClient, version_template: str = None,
                 debug: bool = None, copy_updater: CopyUpdater = None, output_level: str = "essential") -> bool:
    """Create asset or update model metadata if it already exists.

    Args:
        asset (AssetConfig): Asset config.
        registry_name (str): Registry name.
        ml_client (MLClient): MLClient object.
        version_template (str, optional): Version template. Defaults to None.
        debug (bool, optional): Enable debug logging. Defaults to None.
        copy_updater (CopyUpdater, optional): CopyUpdater object to update files during azcopy. Defaults to None.
        output_level (str, optional): Output verbosity level parameter for azcopy. Defaults to "essential".

    Returns:
        bool: True of successfully create/updated, otherwise False.
    """
    # Apply version template
    version = util.apply_version_template(asset.version, version_template)
    logger.print(f"Creating {asset.name} {version}")

    # Update model metadata, if it exists
    if get_asset_details(asset.type.value, asset.name, asset.version, registry_name):
        logger.print(f"{asset.name} {asset.version} already exists, updating the metadata")
        try:
            update_asset_metadata(asset=asset, ml_client=ml_client, allow_no_op_update=False)
            return True
        except Exception as e:
            logger.log_error(f"Failed to update metadata for {asset.name}:{asset.version} - {e}")
            return False

    # Handle specific asset types
    with TemporaryDirectory() as temp_dir:
        if asset.type == AssetType.COMPONENT:
            # load component and check if environment exists
            component_type = asset.spec_as_object().type
            if component_type == ComponentType.PIPELINE.value:
                if not validate_and_prepare_pipeline_component(asset.spec_with_path, registry_name, version_template):
                    return False
            elif component_type is None or component_type in [ComponentType.COMMAND.value,
                                                              ComponentType.PARALLEL.value]:
                if not validate_update_component(asset.spec_with_path, registry_name, version_template):
                    return False
        elif asset.type == AssetType.MODEL:
            version = asset.version
            model_config: ModelConfig = asset.extra_config_as_object()
            if not prepare_model_for_registration(model_config, asset.spec_with_path, Path(temp_dir), ml_client,
                                                  copy_updater, output_level):
                logger.log_error("Failed to prepare model")
                return False
        elif asset.type == AssetType.DATA:
            version = asset.version
            data_config: DataConfig = asset.extra_config_as_object()
            if not prepare_data_for_registration(data_config, asset.spec_with_path, Path(temp_dir), ml_client,
                                                 copy_updater, output_level):
                logger.log_error("Failed to prepare data asset")
                return False

        # Create asset
        return create_asset_cli(
            asset=asset,
            version=version,
            registry_name=registry_name,
            debug=debug,
        )

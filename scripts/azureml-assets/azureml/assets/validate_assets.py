# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Validate assets."""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import List

import azureml.assets as assets
import azureml.assets.util as util
from azureml.assets import PublishLocation, PublishVisibility
from azureml.assets.config import ValidationException
from azureml.assets.util import logger

ERROR_TEMPLATE = "Validation of {asset} failed: {error}"
WARNING_TEMPLATE = "Warning during validation of {asset}: {warning}"

# Common naming convention
NAMING_CONVENTION_URL = "https://github.com/Azure/azureml-assets/wiki/Asset-naming-convention"
INVALID_STRINGS = ["microsoft", ["azureml", "azure"], "aml"]
COMMON_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_.-]{0,254}$")

# Asset config convention
ASSET_CONFIG_URL = "https://github.com/Azure/azureml-assets/wiki/Assets#assetyaml"

# Environment naming convention
ENVIRONMENT_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9.-]{0,254}$")
COMMON_VERSION = r"[0-9]+(?:\.[0-9]+)?(?=-|$)"
FRAMEWORKS = ["pytorch", "sklearn", "tensorflow"]
FRAMEWORK_VERSION = f"(?:{'|'.join(FRAMEWORKS)})-{COMMON_VERSION}"
FRAMEWORK_VERSION_PATTERN = re.compile(FRAMEWORK_VERSION)
INVALID_ENVIRONMENT_STRINGS = ["ubuntu", "cpu"]
OPERATING_SYSTEMS = ["centos", "debian", "win"]
OPERATING_SYSTEM_PATTERN = re.compile(r"(?:centos|debian|\bwin\b)")
OPERATING_SYSTEM_VERSION = f"(?:{'|'.join(OPERATING_SYSTEMS)}){COMMON_VERSION}"
OPERATING_SYSTEM_VERSION_PATTERN = re.compile(OPERATING_SYSTEM_VERSION)
PYTHON_VERSION = r"py3(?:8|9|1[0-9]+)"
PYTHON_VERSION_PATTERN = re.compile(PYTHON_VERSION)
GPU_DRIVERS = ["cuda", "nccl"]
GPU_DRIVER_VERSION = f"(?:{'|'.join(GPU_DRIVERS)}){COMMON_VERSION}"
GPU_DRIVER_VERSION_PATTERN = re.compile(GPU_DRIVER_VERSION)
ENVIRONMENT_NAME_FULL_PATTERN = re.compile("".join([
    FRAMEWORK_VERSION,
    f"(?:-{OPERATING_SYSTEM_VERSION})?",
    f"(?:-{PYTHON_VERSION})?",
    f"(?:-(?:{GPU_DRIVER_VERSION}|gpu))?",
]))

# Image naming convention
IMAGE_NAME_PATTERN = re.compile(r"^azureml/curated/(.+)")


def _log_error(asset_config: assets.AssetConfig, error: str) -> None:
    """Log error.

    Args:
        asset_config (AssetConfig): Asset config.
        error (str): Error message.
    """
    logger.log_error(ERROR_TEMPLATE.format(
        asset=asset_config.spec_with_path,
        error=error
    ))


def _log_warning(asset_config: assets.AssetConfig, warning: str) -> None:
    """Log warning.

    Args:
        asset_config (AssetConfig): Asset config.
        warning (str): Warning message.
    """
    logger.log_warning(WARNING_TEMPLATE.format(
        asset=asset_config.spec_with_path,
        warning=warning
    ))


def validate_environment_name(asset_config: assets.AssetConfig) -> int:
    """Validate environment name.

    Args:
        asset_config (AssetConfig): Asset config.

    Returns:
        int: Number of errors.
    """
    error_count = 0
    asset_name = asset_config.name

    # Check for invalid characters
    if not ENVIRONMENT_NAME_PATTERN.match(asset_name):
        _log_error(asset_config, "Name contains invalid characters")
        error_count += 1

    # Check for invalid strings
    for invalid_string in INVALID_ENVIRONMENT_STRINGS:
        if invalid_string in asset_name:
            _log_error(asset_config, f"Name '{asset_name}' contains invalid string '{invalid_string}'")
            error_count += 1

    # Check for missing frameworks and version
    frameworks_found = [f for f in FRAMEWORKS if f in asset_name]
    if len(frameworks_found) == 0:
        _log_warning(asset_config, f"Name '{asset_name}' is missing framework")
    else:
        # Check framework version
        if not FRAMEWORK_VERSION_PATTERN.search(asset_name):
            _log_error(asset_config, f"Name '{asset_name}' is missing framework version")
            error_count += 1

    # Check operating system and version
    if (OPERATING_SYSTEM_PATTERN.search(asset_name) and
            not OPERATING_SYSTEM_VERSION_PATTERN.search(asset_name)):
        _log_error(asset_config, f"Name '{asset_name}' is missing operating system version")
        error_count += 1

    # Check python version
    if PYTHON_VERSION_PATTERN.search(asset_name):
        _log_warning(asset_config, f"Name '{asset_name}' should only contain Python version if absolutely necessary")

    # Check GPU driver and version
    gpu_drivers_found = [f for f in GPU_DRIVERS if f in asset_name]
    if gpu_drivers_found:
        if not GPU_DRIVER_VERSION_PATTERN.search(asset_name):
            _log_error(asset_config, f"Name '{asset_name}' is missing GPU driver version")
            error_count += 1
        if "gpu" in asset_name:
            _log_error(asset_config, f"Name '{asset_name}' should not contain both GPU driver and 'gpu'")
            error_count += 1

    # Check for ordering
    if frameworks_found and not ENVIRONMENT_NAME_FULL_PATTERN.search(asset_name):
        _log_error(asset_config, f"Name '{asset_name}' elements are out of order")
        error_count += 1

    return error_count


def validate_image_publishing(asset_config: assets.AssetConfig,
                              environment_config: assets.EnvironmentConfig = None) -> int:
    """Validate environment image publishing info.

    Args:
        asset_config (AssetConfig): Asset config.
        environment_config (EnvironmentConfig, optional): Environment config. Defaults to None.

    Returns:
        int: Number of errors.
    """
    error_count = 0
    asset_name = asset_config.name
    environment_config = environment_config or asset_config.extra_config_as_object()

    # Check image name
    if (match := IMAGE_NAME_PATTERN.match(environment_config.image_name)) is not None:
        # Ensure image name matches environment name
        if match.group(1) != asset_name:
            _log_error(asset_config,
                       f"Image name '{environment_config.image_name}' should be 'azureml/curated/{asset_name}'")
            error_count += 1
    else:
        _log_error(asset_config,
                   f"Image name '{environment_config.image_name}' should match pattern azureml/curated/<env_name>")
        error_count += 1

    # Check build context
    if not environment_config.context_dir_with_path.exists():
        _log_error(asset_config,
                   f"Build context directory '{environment_config.context_dir}' not found")
        error_count += 1
    if not environment_config.dockerfile_with_path.exists():
        _log_error(asset_config,
                   f"Dockerfile '{environment_config.dockerfile}' not found")
        error_count += 1

    # Check publishing info
    if not environment_config.publish_enabled:
        _log_error(asset_config, "Image publishing information is not specified")
        error_count += 1

    # Check publishing details
    if environment_config.publish_location != PublishLocation.MCR:
        _log_error(asset_config, "Image publishing location should be 'mcr'")
        error_count += 1
    if environment_config.publish_visibility != PublishVisibility.PUBLIC:
        _log_warning(asset_config, "Image publishing visibility should be 'public'")

    return error_count


def validate_name(asset_config: assets.AssetConfig) -> int:
    """Validate asset name.

    Args:
        asset_config (AssetConfig): Asset config.

    Returns:
        int: Number of errors.
    """
    error_count = 0
    asset_name = asset_config.name

    # Check against generic naming pattern
    if not COMMON_NAME_PATTERN.match(asset_name):
        _log_error(asset_config, f"Name '{asset_name}' contains invalid characters")
        error_count += 1

    # Check for invalid strings
    invalid_strings = INVALID_STRINGS + [asset_config.type.value]
    for string_group in invalid_strings:
        # Coerce into a list
        string_group_list = string_group if isinstance(string_group, list) else [string_group]

        for invalid_string in string_group_list:
            if invalid_string in asset_name:
                # Complain only about the first matching string
                _log_error(asset_config, f"Name '{asset_name}' contains invalid string '{invalid_string}'")
                error_count += 1
                break

    # Validate environment name
    if asset_config.type == assets.AssetType.ENVIRONMENT:
        error_count += validate_environment_name(asset_config)

    return error_count


def validate_categories(asset_config: assets.AssetConfig) -> int:
    """Validate asset categories.

    Args:
        asset_config (AssetConfig): Asset config.

    Returns:
        int: Number of errors.
    """
    if not asset_config.categories:
        _log_error(asset_config, "Categories not found")
        return 1
    return 0


def validate_assets(input_dirs: List[Path],
                    asset_config_filename: str,
                    changed_files: List[Path] = None,
                    check_names: bool = False,
                    check_images: bool = False,
                    check_categories: bool = False) -> bool:
    """Validate assets.

    Args:
        input_dirs (List[Path]): Directories containing assets.
        asset_config_filename (str): Asset config filename to search for.
        changed_files (List[Path], optional): List of changed files, used to filter assets. Defaults to None.
        check_names (bool, optional): Whether to check asset names. Defaults to False.
        check_images (bool, optional): Whether to check image names. Defaults to False.
        check_categories (bool, optional): Whether to check asset categories. Defaults to False.

    Raises:
        ValidationException: If validation fails.

    Returns:
        bool: True if assets were successfully validated, otherwise False.
    """
    # Find assets under input dirs
    asset_count = 0
    error_count = 0
    asset_dirs = defaultdict(list)
    image_names = defaultdict(list)
    for asset_config_path in util.find_asset_config_files(input_dirs, asset_config_filename, changed_files):
        asset_count += 1
        # Load config
        try:
            asset_config = assets.AssetConfig(asset_config_path)
        except Exception as e:
            logger.log_error(f"Validation of {asset_config_path} failed: {e}")
            error_count += 1
            continue
        asset_dirs[f"{asset_config.type.value} {asset_config.name}"].append(asset_config_path)

        # Validate name
        if check_names:
            error_count += validate_name(asset_config)

        # Validate categories
        if check_categories:
            error_count += validate_categories(asset_config)

        # Validate specific asset types
        if asset_config.type == assets.AssetType.ENVIRONMENT:
            try:
                environment_config = asset_config.extra_config_as_object()

                # Store fully qualified image name
                image_name = environment_config.image_name
                if environment_config.publish_location:
                    image_name = f"{environment_config.publish_location.value}/{image_name}"
                image_names[image_name].append(asset_config.file_path)

                # Check image name
                if check_images:
                    error_count += validate_image_publishing(asset_config, environment_config)
            except Exception as e:
                logger.log_error(f"Validation of {asset_config.extra_config_with_path} failed: {e}")
                error_count += 1

        # Validate spec
        try:
            spec = asset_config.spec_as_object()

            # Ensure name and version aren't inconsistent
            if not assets.Config._contains_template(spec.name) and asset_config.name != spec.name:
                raise ValidationException(f"Asset and spec name mismatch: {asset_config.name} != {spec.name}")
            if not assets.Config._contains_template(spec.version) and asset_config.version != spec.version:
                raise ValidationException(f"Asset and spec version mismatch: {asset_config.version} != {spec.version}")
        except Exception as e:
            logger.log_error(f"Validation of {asset_config.spec_with_path} failed: {e}")
            error_count += 1

    # Ensure unique assets
    for type_and_name, dirs in asset_dirs.items():
        if len(dirs) > 1:
            logger.log_error(f"{type_and_name} found in multiple asset YAMLs: {dirs}")
            error_count += 1

    # Ensure unique image names
    for image_name, dirs in image_names.items():
        if len(dirs) > 1:
            logger.log_error(f"{image_name} found in multiple assets: {dirs}")
            error_count += 1

    logger.print(f"Found {error_count} error(s) in {asset_count} asset(s)")
    return error_count == 0


if __name__ == '__main__':
    # Handle command-line args
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-dirs", required=True,
                        help="Comma-separated list of directories containing assets")
    parser.add_argument("-a", "--asset-config-filename", default=assets.DEFAULT_ASSET_FILENAME,
                        help="Asset config file name to search for")
    parser.add_argument("-c", "--changed-files",
                        help="Comma-separated list of changed files, used to filter assets")
    parser.add_argument("-n", "--check-names", action="store_true",
                        help="Check asset names")
    parser.add_argument("-I", "--check-images", action="store_true",
                        help="Check environment images")
    parser.add_argument("-C", "--check-categories", action="store_true",
                        help="Check asset categories")
    args = parser.parse_args()

    # Convert comma-separated values to lists
    input_dirs = [Path(d) for d in args.input_dirs.split(",")]
    changed_files = [Path(f) for f in args.changed_files.split(",")] if args.changed_files else []

    # Share asset naming convention URL
    if args.check_names:
        logger.print(f"Asset naming convention: {NAMING_CONVENTION_URL}")

    # Share asset config reference
    if args.check_categories:
        logger.print(f"Asset config reference: {ASSET_CONFIG_URL}")

    # Validate assets
    success = validate_assets(input_dirs=input_dirs,
                              asset_config_filename=args.asset_config_filename,
                              changed_files=changed_files,
                              check_names=args.check_names,
                              check_images=args.check_images,
                              check_categories=args.check_categories)
    if not success:
        sys.exit(1)

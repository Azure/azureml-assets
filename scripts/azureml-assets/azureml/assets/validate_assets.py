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

ERROR_TEMPLATE = "Validation of {file} failed: {error}"
WARNING_TEMPLATE = "Warning during validation of {file}: {warning}"

# Common naming convention
NAMING_CONVENTION_URL = "https://github.com/Azure/azureml-assets/wiki/Asset-naming-convention"
INVALID_STRINGS = [["azureml", "azure"], "aml"]  # Disallow these in any asset name
NON_MODEL_INVALID_STRINGS = ["microsoft"]  # Allow these in model names
NON_MODEL_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_.-]{0,254}$")

# Asset config convention
ASSET_CONFIG_URL = "https://github.com/Azure/azureml-assets/wiki/Assets#assetyaml"

# Model naming convention
MODEL_NAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,254}$")

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

# Dockerfile convention
DOCKERFILE_IMAGE_PATTERN = re.compile(r"^FROM\s+mcr\.microsoft\.com/azureml/curated/", re.IGNORECASE | re.MULTILINE)

# Image naming convention
IMAGE_NAME_PATTERN = re.compile(r"^azureml/curated/(.+)")

# Docker build context stuff to not allow
BUILD_CONTEXT_DISALLOWED_PATTERNS = [
    re.compile(r"extra-index-url", re.IGNORECASE),
]


def _log_error(file: Path, error: str) -> None:
    """Log error.

    Args:
        file (Path): File with an issue.
        error (str): Error message.
    """
    logger.log_error(ERROR_TEMPLATE.format(
        file=file.as_posix(),
        error=error
    ))


def _log_warning(file: Path, warning: str) -> None:
    """Log warning.

    Args:
        file (Path): File with an issue.
        warning (str): Warning message.
    """
    logger.log_warning(WARNING_TEMPLATE.format(
        file=file.as_posix(),
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
        _log_error(asset_config.file_name_with_path, "Name contains invalid characters")
        error_count += 1

    # Check for invalid strings
    for invalid_string in INVALID_ENVIRONMENT_STRINGS:
        if invalid_string in asset_name:
            _log_error(asset_config.file_name_with_path,
                       f"Name '{asset_name}' contains invalid string '{invalid_string}'")
            error_count += 1

    # Check for missing frameworks and version
    frameworks_found = [f for f in FRAMEWORKS if f in asset_name]
    if len(frameworks_found) == 0:
        _log_warning(asset_config.file_name_with_path, f"Name '{asset_name}' is missing framework")
    else:
        # Check framework version
        if not FRAMEWORK_VERSION_PATTERN.search(asset_name):
            _log_error(asset_config.file_name_with_path, f"Name '{asset_name}' is missing framework version")
            error_count += 1

    # Check operating system and version
    if (OPERATING_SYSTEM_PATTERN.search(asset_name) and
            not OPERATING_SYSTEM_VERSION_PATTERN.search(asset_name)):
        _log_error(asset_config.file_name_with_path,
                   f"Name '{asset_name}' is missing operating system version")
        error_count += 1

    # Check python version
    if PYTHON_VERSION_PATTERN.search(asset_name):
        _log_warning(asset_config.file_name_with_path,
                     f"Name '{asset_name}' should only contain Python version if absolutely necessary")

    # Check GPU driver and version
    gpu_drivers_found = [f for f in GPU_DRIVERS if f in asset_name]
    if gpu_drivers_found:
        if not GPU_DRIVER_VERSION_PATTERN.search(asset_name):
            _log_error(asset_config.file_name_with_path, f"Name '{asset_name}' is missing GPU driver version")
            error_count += 1
        if "gpu" in asset_name:
            _log_error(asset_config.file_name_with_path,
                       f"Name '{asset_name}' should not contain both GPU driver and 'gpu'")
            error_count += 1

    # Check for ordering
    if frameworks_found and not ENVIRONMENT_NAME_FULL_PATTERN.search(asset_name):
        _log_error(asset_config.file_name_with_path, f"Name '{asset_name}' elements are out of order")
        error_count += 1

    return error_count


def validate_dockerfile(environment_config: assets.EnvironmentConfig) -> int:
    """Validate Dockerfile.

    Args:
        environment_config (EnvironmentConfig): Environment config.

    Returns:
        int: Number of errors.
    """
    error_count = 0
    dockerfile = environment_config.get_dockerfile_contents()
    dockerfile = dockerfile.replace("\r\n", "\n")

    if DOCKERFILE_IMAGE_PATTERN.search(dockerfile):
        _log_error(environment_config.dockerfile_with_path,
                   "Referencing curated environment images in Dockerfile is not allowed")
        error_count += 1

    return error_count


def validate_build_context(environment_config: assets.EnvironmentConfig) -> int:
    """Validate environment build context.

    Args:
        environment_config (EnvironmentConfig): Environment config.

    Returns:
        int: Number of errors.
    """
    error_count = 0
    # Iterate over all files in the build context
    for file_path in environment_config.release_paths:
        with open(file_path) as f:
            # Read file into memory, normalize EOL characters
            contents = f.read()
            contents = contents.replace("\r\n", "\n")

            # Check disallowed pattersn
            for pattern in BUILD_CONTEXT_DISALLOWED_PATTERNS:
                if pattern.search(contents):
                    _log_error(file_path, f"Found disallowed pattern '{pattern.pattern}'")
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
            _log_error(environment_config.file_name_with_path,
                       f"Image name '{environment_config.image_name}' should be 'azureml/curated/{asset_name}'")
            error_count += 1
    else:
        _log_error(environment_config.file_name_with_path,
                   f"Image name '{environment_config.image_name}' should match pattern azureml/curated/<env_name>")
        error_count += 1

    # Check build context
    if not environment_config.context_dir_with_path.exists():
        _log_error(environment_config.file_name_with_path,
                   f"Build context directory '{environment_config.context_dir}' not found")
        error_count += 1
    if not environment_config.dockerfile_with_path.exists():
        _log_error(environment_config.file_name_with_path,
                   f"Dockerfile '{environment_config.dockerfile}' not found")
        error_count += 1

    # Check publishing info
    if not environment_config.publish_enabled:
        _log_error(environment_config.file_name_with_path, "Image publishing information is not specified")
        error_count += 1

    # Check publishing details
    if environment_config.publish_location != PublishLocation.MCR:
        _log_error(environment_config.file_name_with_path, "Image publishing location should be 'mcr'")
        error_count += 1
    if environment_config.publish_visibility != PublishVisibility.PUBLIC:
        _log_warning(environment_config.file_name_with_path, "Image publishing visibility should be 'public'")

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
    if not ((asset_config.type is assets.AssetType.MODEL and MODEL_NAME_PATTERN.match(asset_name))
            or NON_MODEL_NAME_PATTERN.match(asset_name)):
        _log_error(asset_config.file_name_with_path, f"Name '{asset_name}' contains invalid characters")
        error_count += 1

    # Check for invalid strings
    invalid_strings = INVALID_STRINGS + [asset_config.type.value]
    if asset_config.type is not assets.AssetType.MODEL:
        invalid_strings += [NON_MODEL_INVALID_STRINGS]
    asset_name_lowercase = asset_name.lower()
    for string_group in invalid_strings:
        # Coerce into a list
        string_group_list = string_group if isinstance(string_group, list) else [string_group]

        for invalid_string in string_group_list:
            if invalid_string in asset_name_lowercase:
                # Complain only about the first matching string
                _log_error(asset_config.file_name_with_path,
                           f"Name '{asset_name}' contains invalid string '{invalid_string}'")
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
        _log_error(asset_config.file_name_with_path, "Categories not found")
        return 1
    return 0


def validate_assets(input_dirs: List[Path],
                    asset_config_filename: str,
                    changed_files: List[Path] = None,
                    check_names: bool = False,
                    check_names_skip_pattern: re.Pattern = None,
                    check_images: bool = False,
                    check_categories: bool = False,
                    check_build_context: bool = False) -> bool:
    """Validate assets.

    Args:
        input_dirs (List[Path]): Directories containing assets.
        asset_config_filename (str): Asset config filename to search for.
        changed_files (List[Path], optional): List of changed files, used to filter assets. Defaults to None.
        check_names (bool, optional): Whether to check asset names. Defaults to False.
        check_names_skip_pattern (re.Pattern, optional): Regex pattern to skip name validation. Defaults to None.
        check_images (bool, optional): Whether to check image names. Defaults to False.
        check_categories (bool, optional): Whether to check asset categories. Defaults to False.
        check_build_context (bool, optional): Whether to check environment build context. Defaults to False.

    Raises:
        ValidationException: If validation fails.

    Returns:
        bool: True if assets were successfully validated, otherwise False.
    """
    # Gather list of just changed assets, for later filtering
    changed_assets = util.find_asset_config_files(input_dirs, asset_config_filename, changed_files) if changed_files else None  # noqa: E501

    # Find assets under input dirs
    asset_count = 0
    error_count = 0
    asset_dirs = defaultdict(list)
    image_names = defaultdict(list)
    for asset_config_path in util.find_asset_config_files(input_dirs, asset_config_filename):
        asset_count += 1
        # Errors only "count" if changed_files was None or the asset was changed
        validate_this = changed_assets is None or asset_config_path in changed_assets

        # Load config
        try:
            asset_config = assets.AssetConfig(asset_config_path)
        except Exception as e:
            if validate_this:
                _log_error(asset_config_path, e)
                error_count += 1
            else:
                _log_warning(asset_config_path, e)
            continue

        # Populate dictionary of asset names to asset config paths
        asset_dirs[f"{asset_config.type.value} {asset_config.name}"].append(asset_config_path)

        # Populate dictionary of image names to asset config paths
        environment_config = None
        if asset_config.type == assets.AssetType.ENVIRONMENT:
            try:
                environment_config = asset_config.extra_config_as_object()

                # Store fully qualified image name
                image_name = environment_config.image_name
                if environment_config.publish_location:
                    image_name = f"{environment_config.publish_location.value}/{image_name}"
                image_names[image_name].append(asset_config.file_path)
            except Exception as e:
                if validate_this:
                    _log_error(environment_config.file_name_with_path, e)
                    error_count += 1
                else:
                    _log_warning(environment_config.file_name_with_path, e)

        # Checks for changed assets only, or all assets if changed_files was None
        if validate_this:
            # Validate name
            if check_names:
                if check_names_skip_pattern is None or not check_names_skip_pattern.fullmatch(asset_config.full_name):
                    error_count += validate_name(asset_config)
                else:
                    logger.log_debug(f"Skipping name validation for {asset_config.full_name}")

            # Validate Dockerfile
            if asset_config.type == assets.AssetType.ENVIRONMENT:
                error_count += validate_dockerfile(asset_config.extra_config_as_object())
                if check_build_context:
                    error_count += validate_build_context(asset_config.extra_config_as_object())

            # Validate categories
            if check_categories:
                error_count += validate_categories(asset_config)

            # Validate specific asset types
            if environment_config is not None:
                if check_images:
                    # Check image name
                    error_count += validate_image_publishing(asset_config, environment_config)

            # Validate spec
            try:
                spec = asset_config.spec_as_object()

                # Ensure name and version aren't inconsistent
                if not assets.Config._contains_template(spec.name) and asset_config.name != spec.name:
                    raise ValidationException(f"Asset and spec name mismatch: {asset_config.name} != {spec.name}")
                if not assets.Config._contains_template(spec.version) and asset_config.version != spec.version:
                    raise ValidationException(f"Asset and spec version mismatch: {asset_config.version} != {spec.version}")  # noqa: E501
            except Exception as e:
                _log_error(asset_config.spec_with_path, e)
                error_count += 1

    # Ensure unique assets
    for type_and_name, dirs in asset_dirs.items():
        if len(dirs) > 1:
            dirs_str = [d.as_posix() for d in dirs]
            logger.log_error(f"{type_and_name} found in multiple asset YAMLs: {dirs_str}")
            error_count += 1

    # Ensure unique image names
    for image_name, dirs in image_names.items():
        if len(dirs) > 1:
            dirs_str = [d.as_posix() for d in dirs]
            logger.log_error(f"{image_name} found in multiple assets: {dirs_str}")
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
    parser.add_argument("-N", "--check-names-skip-pattern", type=re.compile,
                        help="Regex pattern to skip name validation, in the format <type>/<name>/<version>")
    parser.add_argument("-b", "--check-build-context", action="store_true",
                        help="Check environment build context")
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
                              check_names_skip_pattern=args.check_names_skip_pattern,
                              check_images=args.check_images,
                              check_categories=args.check_categories,
                              check_build_context=args.check_build_context)
    if not success:
        sys.exit(1)

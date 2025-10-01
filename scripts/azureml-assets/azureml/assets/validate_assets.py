# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Validate assets."""

import os
import argparse
import json
import re
import sys
import yaml
from collections import defaultdict
from pathlib import Path
from ruamel.yaml import YAML
from typing import List

from azure.ai.ml import load_model
from azure.ai.ml.entities import Model
from azure.ai.ml.operations._run_history_constants import JobStatus
from azure.core.exceptions import ClientAuthenticationError
from azure.identity import AzureCliCredential

import azureml.assets as assets
import azureml.assets.util as util
from azureml.assets import PublishLocation, PublishVisibility
from azureml.assets.config import ValidationException
from azureml.assets.util import logger
from azureml.assets.util.sku_utils import get_all_sku_details

ERROR_TEMPLATE = "Validation of {file} failed: {error}"
WARNING_TEMPLATE = "Warning during validation of {file}: {warning}"

# Common naming convention
NAMING_CONVENTION_URL = "https://github.com/Azure/azureml-assets/wiki/Asset-naming-convention"
INVALID_STRINGS = [["azureml", "azure"], "aml"]  # Disallow these in any asset name
NON_MODEL_INVALID_STRINGS = ["microsoft"]  # Allow these in model names and other assets related to models
MODEL_NAMES_VALIDATION_OVERRIDE_PREFIX = ["Azure-AI-"]  # Exception for Azure-AI models
MODEL_RELATED_ASSETS = [assets.AssetType.MODEL, assets.AssetType.EVALUATIONRESULT]
NON_MODEL_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_.-]{0,254}$")

# Asset config convention
ASSET_CONFIG_URL = "https://github.com/Azure/azureml-assets/wiki/Assets#assetyaml"

# Model naming convention
MODEL_NAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,254}$")

# model validations
MODEL_VALIDATION_RESULTS_FOLDER = "validation_results"
VALIDATION_SUMMARY = "results.json"
SUPPORTED_INFERENCE_SKU_FILE_NAME = "config/supported_inference_skus.json"
SUPPORTED_INFERENCE_SKU_FILE_PATH = Path(__file__).parent / SUPPORTED_INFERENCE_SKU_FILE_NAME


class MLFlowModelProperties:
    """Commonly defined model properties."""

    EVALUATION_RECOMMENDED_SKU = "evaluation-recommended-sku"
    EVALUATION_MIN_SKU_SPEC = "evaluation-min-sku-spec"
    FINETUNE_RECOMMENDED_SKU = "finetune-recommended-sku"
    FINETUNE_MIN_SKU_SPEC = "finetune-min-sku-spec"
    INFERENCE_RECOMMENDED_SKU = "inference-recommended-sku"
    INFERENCE_MIN_SKU_SPEC = "inference-min-sku-spec"

    FINETUNING_TASKS = "finetuning-tasks"
    SHARED_COMPUTE_CAPACITY = "SharedComputeCapacityEnabled"


class MLFlowModelTags:
    """Commonly defined model tags."""

    TASK = "task"
    LICENSE = "license"
    AUTHOR = "author"

    EVALUATION_COMPUTE_ALLOWLIST = "evaluation_compute_allow_list"
    FINETUNE_COMPUTE_ALLOWLIST = "finetune_compute_allow_list"
    FINETUNING_DEFAULTS = "model_specific_defaults"
    INFERENCE_COMPUTE_ALLOWLIST = "inference_compute_allow_list"
    INFERENCE_SUPPORTED_ENVS = "inference_supported_envs"

    # This enables model to use shared quota for deployment
    SHARED_COMPUTE_CAPACITY = "SharedComputeCapacityEnabled"

    # make sure all prod models has hiddenlayerscanned tag added
    # model scan and tag has to be added manually currently
    HIDDEN_LAYERS_SCANNED = "hiddenlayerscanned"


class ModelValidationState:
    """Enums for storing validation results state.

    Keeping it consistent with Job state enums. Only the essential ones would be used here.
    """

    NOT_STARTED = JobStatus.NOT_STARTED
    COMPLETED = JobStatus.COMPLETED
    FAILED = JobStatus.FAILED


class ModelValidationOverallSummary:
    """Enum to capture status of all validations."""

    BATCH_DEPLOYMENT = "BatchDeployment"
    ONLINE_DEPLOYMENT = "OnlineDeployment"
    VALIDATION_RUN = "ValidationRun"
    BUILD_URI = "BuildUri"

    @staticmethod
    def get_default_summary():
        """Return model validation default summary."""
        return {
            ModelValidationOverallSummary.BATCH_DEPLOYMENT: ModelValidationState.NOT_STARTED,
            ModelValidationOverallSummary.BATCH_ONLINE_DEPLOYMENTDEPLOYMENT: ModelValidationState.NOT_STARTED,
            ModelValidationOverallSummary.VALIDATION_RUN: ModelValidationState.NOT_STARTED,
        }


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

# Directory path for config files related to asset validation, relative to the current source path
CONFIG_DIRECTORY = "config"


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


def validate_environment_version(asset_config: assets.AssetConfig) -> int:
    """Validate environment version.

    Args:
        asset_config (AssetConfig): Asset config.

    Returns:
        int: Number of errors.
    """
    if not asset_config.auto_version:
        _log_error(asset_config.file_name_with_path,
                   f"Environment version must be auto but is {asset_config.version}")
        return 1

    return 0


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
    if asset_config.type not in MODEL_RELATED_ASSETS:
        invalid_strings += [NON_MODEL_INVALID_STRINGS]
    asset_name_lowercase = asset_name.lower()
    for string_group in invalid_strings:
        # Coerce into a list
        string_group_list = string_group if isinstance(string_group, list) else [string_group]

        for invalid_string in string_group_list:
            if invalid_string in asset_name_lowercase:
                if (asset_config.type == assets.AssetType.MODEL and
                        any(asset_name.startswith(exception) for exception in MODEL_NAMES_VALIDATION_OVERRIDE_PREFIX)):
                    continue
                # Complain only about the first matching string
                _log_error(asset_config.file_name_with_path,
                           f"Name '{asset_name}' contains invalid string '{invalid_string}'")
                error_count += 1
                break

    # Validate environment name
    if asset_config.type == assets.AssetType.ENVIRONMENT:
        error_count += validate_environment_name(asset_config)

    return error_count


def validate_tests(asset_config: assets.AssetConfig) -> int:
    """Validate pytest information.

    Args:
        asset_config (assets.AssetConfig): Asset config.

    Returns:
        int: Number of errors.
    """
    error_count = 0

    # Bail early if pytest is not enabled
    if not asset_config.pytest_enabled:
        return error_count

    # Check file/dir references
    if not asset_config.pytest_tests_dir_with_path.exists():
        _log_error(asset_config.file_name_with_path,
                   f"pytest.tests_dir directory '{asset_config.pytest_tests_dir}' not found")
        error_count += 1
    if asset_config.pytest_conda_environment and not asset_config.pytest_conda_environment_with_path.exists():
        _log_error(asset_config.file_name_with_path,
                   f"pytest.conda_environment file '{asset_config.pytest_conda_environment}' not found")
        error_count += 1
    if asset_config.pytest_pip_requirements and not asset_config.pytest_pip_requirements_with_path.exists():
        _log_error(asset_config.file_name_with_path,
                   f"pytest.pip_requirements file '{asset_config.pytest_pip_requirements}' not found")
        error_count += 1

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


def validate_tags(asset_config: assets.AssetConfig, valid_tags_filename: str) -> int:
    """Validate proper tags for an asset.

    Args:
        asset_config (AssetConfig): Asset config.
        valid_tags_filename (str): Yaml filename in the config directory defining valid tags for the asset.

    Returns:
        int: Number of errors.
    """
    error_count = 0

    with open(Path(__file__).parent / CONFIG_DIRECTORY / valid_tags_filename) as f:
        valid_tags = YAML().load(f)

    asset_spec = asset_config._spec._yaml
    asset_tags = asset_spec.get('tags')

    for tag in valid_tags:
        tag_info = valid_tags[tag]
        if tag_info.get('required') and (not asset_tags or tag not in asset_tags):
            _log_error(asset_config.file_name_with_path,
                       f"Asset '{asset_config.name}' is missing required tag '{tag}'")
            error_count += 1
            continue

        if not asset_tags or tag not in asset_tags:
            continue

        tag_value = asset_tags[tag]
        if not isinstance(tag_value, str):
            _log_error(asset_config.file_name_with_path,
                       f"Asset '{asset_config.name}' has non-string '{type(tag_value)}' for tag '{tag}'")
            error_count += 1
            continue

        valid_tag_values = tag_info.get('values')
        if valid_tag_values:
            if tag_info.get('allow_multiple'):
                tag_values = tag_value.split(',')
                for single_tag_value in tag_values:
                    if single_tag_value not in valid_tag_values:
                        _log_error(asset_config.file_name_with_path,
                                   f"Asset '{asset_config.name}' has invalid value '{single_tag_value}' for tag '{tag}'. Valid values are {valid_tag_values}")  # noqa: E501
                        error_count += 1
                        continue
            else:
                if tag_value not in valid_tag_values:
                    _log_error(asset_config.file_name_with_path,
                               f"Asset '{asset_config.name}' has invalid value '{tag_value}' for tag '{tag}'. Valid values are {valid_tag_values}")  # noqa: E501
                    error_count += 1
                    continue

    return error_count


def confirm_model_validation_results(
    latest_asset_config: assets.AssetConfig,
    validated_asset_config: assets.AssetConfig
) -> int:
    """Compare latest model with validation results.

    Args:
        latest_asset_config (assets.AssetConfig): asset config for latest model
        validated_asset_config (assets.AssetConfig): asset cofig for validated model

    Returns:
        int: Number of errors.

    """
    error_count = 0
    try:
        latest_model_config: assets.ModelConfig = latest_asset_config.extra_config_as_object()
        if latest_model_config.type != assets.config.ModelType.MLFLOW:
            logger.print(
                f"Bypass validation for {latest_asset_config.name} as model type is: {latest_model_config.type.value}"
            )
            return 0

        if not validated_asset_config:
            logger.log_error(f"Validated asset config is None for {latest_asset_config.name}")
            return 1

        validated_model_config: assets.ModelConfig = validated_asset_config.extra_config_as_object()
        logger.print(f"Comparing validated and latest model asset files for {latest_asset_config.name}")

        latest_model_path_uri = latest_model_config.path.uri
        if (latest_model_config.path.type == assets.PathType.AZUREBLOB):
            latest_model_path_uri = latest_model_config.path.uri.split("?")[0]

        validated_model_path_uri = validated_model_config.path.uri
        if (latest_model_config.path.type == assets.PathType.AZUREBLOB):
            validated_model_path_uri = validated_model_config.path.uri.split("?")[0]

        if not (
            latest_model_config.path.type == validated_model_config.path.type and
            latest_model_path_uri == validated_model_path_uri and
            latest_model_config.description == validated_model_config.description and
            latest_model_config.type == validated_model_config.type
        ):
            logger.log_error(
                "Validated model config does not match with latest model. "
                "Either last validation run for model had failed or its still running."
            )

            if latest_model_config.type != validated_model_config.type:
                logger.log_warning(
                    f"latest_model_config_type: [{latest_model_config.type}] and "
                    f"validated_model_config_type: [{validated_model_config.type}] does not match."
                )
                error_count += 1

            if latest_model_config.path.type != validated_model_config.path.type:
                logger.log_warning(
                    f"latest_model_config_path_type: [{latest_model_config.path.type}] and "
                    f"validated_model_config_path_type: [{validated_model_config.path.type}] does not match."
                )
                error_count += 1

            if latest_model_path_uri != validated_model_path_uri:
                logger.log_warning(
                    f"latest_model_config_path_uri: [{latest_model_path_uri}] and "
                    f"validated_model_config_path_uri: [{validated_model_path_uri}] does not match."
                )
                error_count += 1

            if latest_model_config.description != validated_model_config.description:
                logger.log_warning("Description does not match, for latest and validated asset")
                error_count += 1

        # check if spec has changes
        latest_model = load_model(latest_asset_config.spec_with_path)
        validated_model = load_model(validated_asset_config.spec_with_path)

        if latest_model.version != validated_model.version:
            logger.log_error("version mismatch")
            logger.log_warning(f"latest_model tags: [[{latest_model.version}]]")
            logger.log_warning(f"validated_model: [[{validated_model.version}]]")
            error_count += 1

        if latest_model.tags != validated_model.tags:
            logger.log_error("tags mismatch")
            logger.log_warning(f"latest_model tags: [{latest_model.tags}]")
            logger.log_warning(f"validated_model: [{validated_model.tags}]")
            error_count += 1

        if latest_model.properties != validated_model.properties:
            logger.log_error("properties mismatch")
            logger.log_warning(f"latest_model properties: [{latest_model.properties}]")
            logger.log_warning(f"validated_model properties: [{validated_model.properties}]")
            error_count += 1

        if latest_model.description != validated_model.description:
            logger.log_error("description mismatch")
            logger.log_warning(f"latest_model description: [{latest_model.description}]")
            logger.log_warning(f"validated_model description: [{validated_model.description}]")
            error_count += 1

        # check validation results now
        validation_results_dir = validated_asset_config.file_path / MODEL_VALIDATION_RESULTS_FOLDER
        validation_job_details_path = validation_results_dir / VALIDATION_SUMMARY
        if not validation_job_details_path.exists():
            logger.log_error(
                f"{VALIDATION_SUMMARY} missing for model {latest_asset_config.name}. "
                "Either last validation run for model had failed or its still running."
            )
            error_count += 1
        else:
            overall_summary = {}
            with open(validation_job_details_path) as f:
                overall_summary = json.load(f)

            validation_run_status = overall_summary.get(
                ModelValidationOverallSummary.VALIDATION_RUN, ModelValidationState.NOT_STARTED)
            batch_deployment_status = overall_summary.get(
                ModelValidationOverallSummary.BATCH_DEPLOYMENT, ModelValidationState.NOT_STARTED)
            online_deployment_status = overall_summary.get(
                ModelValidationOverallSummary.ONLINE_DEPLOYMENT, ModelValidationState.NOT_STARTED)
            buildUri = overall_summary.get(ModelValidationOverallSummary.BUILD_URI, None)

            if validation_run_status != ModelValidationState.COMPLETED:
                logger.log_error(
                    f"run status for model {latest_asset_config.name} is {validation_run_status}. "
                    "Please ensure that there is a completed model validation job."
                )
                error_count += 1

            # check is batch supported?
            disable_batch_str = latest_model.tags.get("disable-batch", "false")
            supports_batch = True if disable_batch_str.lower() == "false" else False

            if supports_batch and batch_deployment_status != ModelValidationState.COMPLETED:
                logger.log_error(
                    f"batch deployment is supported for {latest_model.name}. "
                    f"But its status is {batch_deployment_status}. "
                    "Please ensure that that batch is validated for the model."
                )
                error_count += 1

            # check if online inference is supported
            supports_inference = (
                True
                if latest_model.tags.get(MLFlowModelTags.INFERENCE_COMPUTE_ALLOWLIST, None)
                or latest_model.properties.get(MLFlowModelProperties.INFERENCE_RECOMMENDED_SKU)
                else False
            )

            if supports_inference and online_deployment_status != ModelValidationState.COMPLETED:
                logger.log_error(
                    f"Online deployment is supported for {latest_model.name}. "
                    f"But its status is {online_deployment_status}. "
                    "Please ensure that that online inference is validated for the model."
                )
                error_count += 1

            if error_count > 0 and buildUri:
                logger.print(f"Check model: {latest_model.name} validation logs here: {buildUri}")

        return error_count
    except Exception as e:
        logger.log_error(
            f"Exception when confirming validation results for model {latest_asset_config.name}. Exception {e}"
        )
        return 1


def validate_model_scenario(
    asset_file_name_with_path: Path,
    model: Model,
    min_sku_prop_name: str,
    recommended_skus_prop_name: str,
    compute_allowlist_tags_name: str,
) -> int:
    """Validate model properties, tags for different scenarios.

    Args:
        asset_file_name_with_path (Path): file path to model asset
        model (Model): model loaded from spec
        min_sku_prop_name (str): min sku property name for the scenario
        recommended_skus_prop_name (str): recommended sku property name for the scenario
        compute_allowlist_tags_name (str): compute allowlist tag name for the scenario

    Returns:
        int: Number of errors.

    """
    error_count = 0
    min_sku = model.properties.get(min_sku_prop_name, "").strip()
    recommended_skus_value = model.properties.get(recommended_skus_prop_name, None)
    if isinstance(recommended_skus_value, str):
        recommended_skus = [sku.strip() for sku in recommended_skus_value.split(",") if sku.strip()]
    elif isinstance(recommended_skus_value, list):
        recommended_skus = [str(sku).strip() for sku in recommended_skus_value if str(sku).strip()]
    else:
        recommended_skus = []
    compute_allowlists = set(model.tags.get(compute_allowlist_tags_name, []))

    if not min_sku:
        _log_error(asset_file_name_with_path, f"{min_sku_prop_name} is missing in model properties")
        error_count += 1

    if not recommended_skus:
        _log_error(asset_file_name_with_path, f"{recommended_skus_prop_name} is missing in model properties")
        error_count += 1

    if not compute_allowlists:
        _log_error(asset_file_name_with_path, f"{compute_allowlist_tags_name} is missing in model tags")
        error_count += 1

    recommended_skus_set = set(recommended_skus)
    if (recommended_skus_set != compute_allowlists):
        a_minus_b = recommended_skus_set - compute_allowlists
        b_minus_a = compute_allowlists - recommended_skus_set
        _log_error(
            asset_file_name_with_path,
            f"{recommended_skus_prop_name} and {compute_allowlist_tags_name} does not match for model.\n"
            f"skus in recommended sku and not in compute allowlists => {a_minus_b}\n"
            f"skus in compute allowlists and not in recommended sku => {b_minus_a}\n"
        )
        error_count += 1

    # confirm min_sku_spec with list of supported computes
    error_count += confirm_min_sku_spec(asset_file_name_with_path, min_sku_prop_name, compute_allowlists, min_sku)

    return error_count


def confirm_min_sku_spec(
    asset_file_name_with_path: Path,
    min_sku_prop_name: str,
    supported_skus: set,
    min_sku_spec: str
):
    """Validate model properties, tags for different scenarios.

    Args:
        asset_file_name_with_path (Path): file path to model asset
        min_sku_prop_name (str): min sku property name for the scenario
        supported_skus (List): supported SKUs for the scenario
        min_sku_spec (str): Scenario min SKU spec

    Returns:
        int: Number of errors.
    """
    subscription_id = os.getenv("SUBSCRIPTION_ID")
    if not subscription_id:
        logger.log_warning("subscription_id missing. Skipping min sku valdn")
        return 0

    try:
        all_sku_details = get_all_sku_details(AzureCliCredential(), subscription_id)
        min_disk = min_cpu_mem = min_ngpus = min_ncpus = -1
        for sku in supported_skus:
            sku_details = all_sku_details.get(sku)
            if not sku_details:
                raise Exception(
                    f"Caught exception while checking {min_sku_prop_name}."
                    f" Either invalid sku {sku} or issue with fetching sku details"
                )

            num_cpus = sku_details["vCPUs"]
            num_gpus = sku_details["gpus"]
            cpu_mem = int(sku_details["memoryGB"])
            disk_space = int(sku_details["maxResourceVolumeMB"] / 1024)

            min_ncpus = min(num_cpus, min_ncpus) if min_ncpus > 0 else num_cpus
            min_ngpus = min(num_gpus, min_ngpus) if min_ngpus >= 0 else num_gpus
            min_cpu_mem = min(cpu_mem, min_cpu_mem) if min_cpu_mem > 0 else cpu_mem
            min_disk = min(disk_space, min_disk) if min_disk > 0 else disk_space

        ncpus, ngpus, mem, disk = [int(item) for item in min_sku_spec.split("|")]
        if ncpus != min_ncpus or ngpus != min_ngpus or mem != min_cpu_mem or disk != min_disk:
            _log_error(
                asset_file_name_with_path,
                f"for {min_sku_prop_name} => "
                f"{ncpus}|{ngpus}|{mem}|{disk} != {min_ncpus}|{min_ngpus}|{min_cpu_mem}|{min_disk}"
            )

            # list of skus larger than current specific min-sku
            skus_failing_valdn = []
            for sku in supported_skus:
                sku_details = all_sku_details.get(sku)
                num_cpus = sku_details["vCPUs"]
                num_gpus = sku_details["gpus"]
                cpu_mem = int(sku_details["memoryGB"])
                disk_space = int(sku_details["maxResourceVolumeMB"] / 1024)

                if num_cpus < ncpus or num_gpus < ngpus or cpu_mem < mem or disk_space < disk:
                    sku_spec = "|".join([str(num_cpus), str(num_gpus), str(cpu_mem), str(disk_space)])
                    skus_failing_valdn.append(f"{sku}: {sku_spec}")

            _log_error(
                asset_file_name_with_path,
                f"for {min_sku_prop_name} => "
                f"SKUs having smaller spec: {skus_failing_valdn}"
            )

            return 1
    except ClientAuthenticationError as e:
        logger.log_warning(f"Failed to log in to Azure, skipping min sku valdn. {e}")
    except Exception as e:
        _log_error(asset_file_name_with_path, f"Exception in fetching SKU details: {e}")
        return 1
    return 0


def validate_model_spec(asset_config: assets.AssetConfig) -> int:
    """Validate model spec.

    Args:
        asset_config (assets.AssetConfig): asset config for model spec

    Returns:
        int: error count
    """
    error_count = 0
    model = model_config = None

    try:
        model = load_model(asset_config.spec_with_path)
    except Exception:
        _log_error(asset_config.file_name_with_path, "Invalid spec file")
        return 1

    try:
        model_config: assets.ModelConfig = asset_config.extra_config_as_object()
    except Exception:
        _log_error(asset_config.file_name_with_path, "Invalid model config")
        return 1

    if model_config.type != assets.config.ModelType.MLFLOW:
        logger.print(
            f"Bypass validation for {asset_config.name} as model type is: {model_config.type.value}"
        )
        return 0

    # confirm must have
    if not model.tags.get(MLFlowModelTags.TASK):
        _log_error(asset_config.file_name_with_path, f"{MLFlowModelTags.TASK} missing")
        error_count += 1

    if not model.tags.get(MLFlowModelTags.LICENSE):
        _log_error(asset_config.file_name_with_path, f"{MLFlowModelTags.LICENSE} missing")
        error_count += 1

    # TODO: skip hidden layer valdn till most models are scanned
    # if MLFlowModelTags.HIDDEN_LAYERS_SCANNED not in model.tags:
    #     _log_error(
    #         asset_config.file_name_with_path,
    #         (
    #             "`hiddenlayerscanned` tag missing. "
    #             "Model is not scanned by HiddenLayer ModelScanner tool. Please scan the model and retry"
    #         )
    #     )
    #     error_count += 1

    # shared compute check
    if MLFlowModelTags.SHARED_COMPUTE_CAPACITY not in model.tags:
        _log_error(asset_config.file_name_with_path, f"Tag {MLFlowModelTags.SHARED_COMPUTE_CAPACITY} missing")
        error_count += 1

    if not model.properties.get(MLFlowModelProperties.SHARED_COMPUTE_CAPACITY, False):
        _log_error(
            asset_config.file_name_with_path, f"Property {MLFlowModelProperties.SHARED_COMPUTE_CAPACITY} is not set"
        )
        error_count += 1

    # if any of the relevant tags or properies is present
    # assume support as true and then fail in valdn
    supports_eval = (
        MLFlowModelTags.EVALUATION_COMPUTE_ALLOWLIST in model.tags
        or MLFlowModelProperties.EVALUATION_RECOMMENDED_SKU in model.properties
        or MLFlowModelProperties.EVALUATION_MIN_SKU_SPEC in model.properties
    )

    # If any of the relevant tags or properies is present
    # assume support as true and then fail in valdn
    supports_ft = (
        MLFlowModelTags.FINETUNE_COMPUTE_ALLOWLIST in model.tags
        or MLFlowModelProperties.FINETUNE_RECOMMENDED_SKU in model.properties
        or MLFlowModelProperties.FINETUNE_MIN_SKU_SPEC in model.properties
        or MLFlowModelProperties.FINETUNING_TASKS in model.properties
    )

    # validate inference compute req.
    error_count += validate_model_scenario(
        asset_config.file_name_with_path,
        model,
        MLFlowModelProperties.INFERENCE_MIN_SKU_SPEC,
        MLFlowModelProperties.INFERENCE_RECOMMENDED_SKU,
        MLFlowModelTags.INFERENCE_COMPUTE_ALLOWLIST,
    )

    # check valid computes for inference
    with open(SUPPORTED_INFERENCE_SKU_FILE_PATH) as f:
        supported_inference_skus = set(json.load(f))
        unsupported_skus_in_spec = [
            sku
            for sku in model.tags.get(MLFlowModelTags.INFERENCE_COMPUTE_ALLOWLIST, [])
            if sku not in supported_inference_skus
        ]
        if unsupported_skus_in_spec:
            _log_error(asset_config.file_name_with_path,
                       f"Unsupported inference SKU in spec: {unsupported_skus_in_spec}")
            error_count += 1

    if supports_eval:
        error_count += validate_model_scenario(
            asset_config.file_name_with_path,
            model,
            MLFlowModelProperties.EVALUATION_MIN_SKU_SPEC,
            MLFlowModelProperties.EVALUATION_RECOMMENDED_SKU,
            MLFlowModelTags.EVALUATION_COMPUTE_ALLOWLIST,
        )

    if supports_ft:
        if not model.properties.get(MLFlowModelProperties.FINETUNING_TASKS):
            _log_error(
                asset_config.file_name_with_path,
                f"{MLFlowModelProperties.FINETUNING_TASKS} not set for supporting finetuning scenario"
            )
            error_count += 1

        error_count += validate_model_scenario(
            asset_config.file_name_with_path,
            model,
            MLFlowModelProperties.FINETUNE_MIN_SKU_SPEC,
            MLFlowModelProperties.FINETUNE_RECOMMENDED_SKU,
            MLFlowModelTags.FINETUNE_COMPUTE_ALLOWLIST,
        )

    return error_count


def validate_mlflow_model(asset_config: assets.AssetConfig) -> int:
    """Validate if model is of type MLFlow.

    Args:
        asset_config (assets.AssetConfig): asset config for model spec
    """
    model_config = None

    try:
        load_model(asset_config.spec_with_path)
        model_config: assets.ModelConfig = asset_config.extra_config_as_object()
    except Exception:
        logger.log_warning("Could not validate if model is of type MLFlow due to invalid spec or model config")
        return

    if model_config.type == assets.config.ModelType.MLFLOW:
        logger.log_warning(f"{asset_config.name} is a model of type {model_config.type.value} "
                           f"which is banned from the model catalog")

        # Update mlflow_model_detected variable for Github Actions only
        if "GITHUB_OUTPUT" in os.environ:
            logger.print("Setting GITHUB_OUTPUT env variable for mlflow_model_detected=true")
            with open(os.environ["GITHUB_OUTPUT"], "a") as f:
                print("mlflow_model_detected=true", file=f)


def get_validated_models_assets_map(model_validation_results_dir: str):
    """Return model assets map."""
    try:
        if not model_validation_results_dir:
            logger.log_warning(
                "Unexpected !!! model_validation_results_dir is None. Model assets might fail in validation."
            )
            return {}

        validated_model_assets: List[assets.AssetConfig] = util.find_assets(
            input_dirs=[Path(model_validation_results_dir)],
            asset_config_filename=assets.DEFAULT_ASSET_FILENAME,
            types=[assets.config.AssetType.MODEL]
        )

        return {model_asset.name: model_asset for model_asset in validated_model_assets}
    except Exception as e:
        logger.log_error(f"Error in creating validated model map => {e}")
        return {}


def validate_assets(input_dirs: List[Path],
                    asset_config_filename: str,
                    model_validation_results_dir: str = None,
                    changed_files: List[Path] = None,
                    check_names: bool = False,
                    check_names_skip_pattern: re.Pattern = None,
                    check_images: bool = False,
                    check_categories: bool = False,
                    check_build_context: bool = False,
                    check_tests: bool = False,
                    check_environment_version: bool = False,
                    added_files: List[Path] = None) -> bool:
    """Validate assets.

    Args:
        input_dirs (List[Path]): Directories containing assets.
        asset_config_filename (str): Asset config filename to search for.
        model_validation_results_dir (str, optional): Dir containing model validation results
        changed_files (List[Path], optional): List of changed files, used to filter assets. Defaults to None.
        check_names (bool, optional): Whether to check asset names. Defaults to False.
        check_names_skip_pattern (re.Pattern, optional): Regex pattern to skip name validation. Defaults to None.
        check_images (bool, optional): Whether to check image names. Defaults to False.
        check_categories (bool, optional): Whether to check asset categories. Defaults to False.
        check_build_context (bool, optional): Whether to check environment build context. Defaults to False.
        check_tests (bool, optional): Whether to check test references. Defaults to False.
        check_environment_version (bool, optional): Whether to check environment version. Defaults to False.
        added_files (List[Path], optional): List of added files, used for validating new assets. Defaults to None.

    Raises:
        ValidationException: If validation fails.

    Returns:
        bool: True if assets were successfully validated, otherwise False.
    """
    # Gather list of just changed assets, for later filtering
    changed_assets = util.find_asset_config_files(input_dirs, asset_config_filename, changed_files) if changed_files else None  # noqa: E501
    added_assets = util.find_asset_config_files(input_dirs, asset_config_filename, added_files) if added_files else None  # noqa: E501
    validated_model_map = get_validated_models_assets_map(model_validation_results_dir)

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

        # Extract model variant info and defaultDeploymentTemplate from spec (not supported in SDK)
        unsupported_fields = ["variantInfo", "defaultDeploymentTemplate"]
        unsupported_fields_mapping = {}

        if asset_config.type == assets.AssetType.MODEL:
            with open(asset_config.spec_with_path, "r") as f:
                spec_config = yaml.safe_load(f)

                for field in unsupported_fields:
                    if field in spec_config:
                        logger.print(f"Found unsupported SDK field '{field}' in spec for asset {asset_config.name}")
                        unsupported_fields_mapping[field] = spec_config.pop(field)

            if unsupported_fields_mapping:
                logger.print(f"Found unsupported fields in spec, popping out info and rewriting spec. "
                             f"unsupported fields: {unsupported_fields_mapping}")
                with open(asset_config.spec_with_path, "w") as f:
                    yaml.dump(spec_config, f)

        # Populate dictionary of asset names to asset config paths
        asset_dirs[f"{asset_config.type.value} {asset_config.name}"].append(asset_config_path)

        # validated_model_map would be empty for non-drop scenario
        if asset_config.type == assets.AssetType.MODEL:
            error_count += validate_model_spec(asset_config)
            # should run during drop creation only
            if validated_model_map:
                error_count += confirm_model_validation_results(
                    asset_config,
                    validated_model_map.get(asset_config.name, None)
                )

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

            # Validate pytest information
            if check_tests:
                error_count += validate_tests(asset_config)

            # Validate if MLFlow model if new asset
            if asset_config.type == assets.AssetType.MODEL:
                if added_assets and asset_config_path in added_assets:
                    logger.print(f"Validating type of new model: {asset_config.name}")
                    validate_mlflow_model(asset_config)

            if asset_config.type == assets.AssetType.ENVIRONMENT:
                # Validate Dockerfile
                error_count += validate_dockerfile(asset_config.extra_config_as_object())
                if check_build_context:
                    error_count += validate_build_context(asset_config.extra_config_as_object())

                # Validate environment version
                if check_environment_version:
                    error_count += validate_environment_version(asset_config)

            if asset_config.type == assets.AssetType.EVALUATIONRESULT:
                error_count += validate_tags(asset_config, 'evaluationresult/tag_values_shared.yaml')

                asset_spec = asset_config._spec._yaml
                evaluation_type = asset_spec.get('tags', {}).get('evaluation_type', None)

                evaluation_tag_files = {
                    'text_generation': 'evaluationresult/tag_values_text_generation.yaml',
                    'text_embeddings': 'evaluationresult/tag_values_text_embeddings.yaml',
                    'vision': 'evaluationresult/tag_values_vision.yaml',
                    'text_quality': 'evaluationresult/tag_values_text_quality.yaml',
                    'text_performance': 'evaluationresult/tag_values_text_performance.yaml',
                    'text_cost': 'evaluationresult/tag_values_text_cost.yaml'
                }

                if evaluation_type in evaluation_tag_files:
                    error_count += validate_tags(asset_config, evaluation_tag_files[evaluation_type])
                else:
                    _log_error(
                        asset_config.file_name_with_path,
                        f"Asset '{asset_config.name}' has unknown evaluation_type: '{evaluation_type}'"
                    )
                    error_count += 1

            if asset_config.type == assets.AssetType.PROMPT:
                error_count += validate_tags(asset_config, 'tag_values_shared.yaml')
                error_count += validate_tags(asset_config, 'tag_values_prompt.yaml')

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

        # Write unsupported fields back to spec
        if unsupported_fields_mapping:
            spec_config.update(unsupported_fields_mapping)
            with open(asset_config.spec_with_path, "w") as f:
                yaml.dump(spec_config, f)

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
    parser.add_argument("-m", "--model-validation-results-dir", required=False,
                        help="Dir containing model validation results and validated spec.")
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
    parser.add_argument("-t", "--check-tests", action="store_true",
                        help="Check test references")
    parser.add_argument("-e", "--check-environment-version", action="store_true",
                        help="Check environment version")
    parser.add_argument("-d", "--added-files",
                        help="Comma-separated list of added files, used to validate new assets")
    args = parser.parse_args()

    # Convert comma-separated values to lists
    input_dirs = [Path(d) for d in args.input_dirs.split(",")]
    changed_files = [Path(f) for f in args.changed_files.split(",")] if args.changed_files else []
    added_files = [Path(f) for f in args.added_files.split(",")] if args.added_files else []

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
                              check_build_context=args.check_build_context,
                              model_validation_results_dir=args.model_validation_results_dir,
                              check_tests=args.check_tests,
                              check_environment_version=args.check_environment_version,
                              added_files=added_files)
    if not success:
        sys.exit(1)

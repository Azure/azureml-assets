# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Python script to publish assets."""

import argparse
import os
import re
import shutil
import sys
from pathlib import Path
from string import Template
from subprocess import PIPE, STDOUT, run
from tempfile import TemporaryDirectory
from typing import List
import azureml.assets as assets
import azureml.assets.util as util
import yaml
from azureml.assets.config import PathType
from azureml.assets.release.model_publish_utils import ModelDownloadUtils
from azureml.assets.util import logger

ASSET_ID_TEMPLATE = Template(
    "azureml://registries/$registry_name/$asset_type/$asset_name/versions/$version")
TEST_YML = "tests.yml"


def test_files_location(dir: Path):
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


def update_model_spec_file(spec_file: Path, path: Path):
    """Update the yaml file after getting the model has been prepared."""
    try:
        with open(spec_file) as f:
            model_file = yaml.safe_load(f)
        model_file['path'] = path
        with open(spec_file, "w") as f:
            yaml.dump(model_file, f)
    except Exception as e:
        logger.print(f"Error while updating spec")
        raise e


def model_prepare(
    model_config: assets.ModelConfig, spec_file: Path, model_dir: Path
) -> bool:
    """Prepare Model."""
    """
    Prepare the model. Download the model if required.
    Convert the models to specified publish type.

    Return: returns the local path to the model.
    """

    if model_config.path.type == PathType.LOCAL:
        update_model_spec_file(spec_file, os.path.abspath(
            Path(model_config.path.uri).resolve()))
        return True

    if model_config.type == assets.ModelType.CUSTOM:
        validate_download = ModelDownloadUtils.download_model(model_config.path.type, model_config.path.uri, model_dir)
        if validate_download:
            update_model_spec_file(spec_file, model_dir)

    elif model_config.type == assets.ModelType.MLFLOW:
        # TODO: udpate this once we start consuming git based model
        validate_download = ModelDownloadUtils.download_model(model_config.path.type, model_config.path.uri, model_dir)
        if validate_download:
            update_model_spec_file(spec_file, model_dir)

    else:
        print(model_config.type.value, assets.ModelType.MLFLOW)
        validate_download = False
        logger.log_error(f"Model type {model_config.type} not supported yet")

    return validate_download


def assemble_command(
    asset_type: str,
    # subscription_id: str,
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
        # "--subscription", subscription_id,
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


def run_command(
    cmd, failure_list: List, debug_mode: bool = None
):
    """Run the az cli command for pushing the model to registry."""
    if debug_mode:
        # Capture and redact output
        results = run(
            cmd,
            stdout=PIPE,
            stderr=STDOUT,
            encoding=sys.stdout.encoding,
            errors="ignore",
        )
        redacted_output = re.sub(r"Bearer.*", "", results.stdout)
        logger.print(redacted_output)
    else:
        results = run(cmd)

    # Check command result
    if results.returncode != 0:
        logger.log_warning(f"Error creating {asset}")
        failure_list.append(asset)


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
    for asset in util.find_assets(input_dirs=assets_dir):
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
        if asset.type in [assets.AssetType.COMPONENT, assets.AssetType.ENVIRONMENT]:
            # Assemble command
            cmd = assemble_command(
                asset.type.value, str(asset.spec_with_path),
                registry_name, final_version, resource_group, workspace, debug_mode)
            # Run command
            run_command(cmd, failure_list, debug_mode)

        elif asset.type == assets.AssetType.MODEL:

            model_config = asset.extra_config_as_object()

            with TemporaryDirectory() as tempdir:
                result = model_prepare(
                    model_config, asset.spec_with_path, tempdir)
                # Run command
                if result:
                    # Assemble Command
                    cmd = assemble_command(
                        asset.type.value, str(asset.spec_with_path),
                        registry_name, final_version, resource_group, workspace, debug_mode)
                    run_command(cmd, failure_list, debug_mode)

        else:
            logger.log_warning(f"unsupported asset type: {asset.type.value}")

    if len(failure_list) > 0:
        logger.log_warning(
            f"following assets failed to publish: {failure_list}")

    if tests_dir:
        logger.print("locating test files")
        test_jobs = test_files_location(tests_dir)

        logger.print("preprocessing test files")
        preprocess_test_files(test_jobs, asset_ids)
        logger.print("finished preprocessing test files")
    else:
        logger.log_warning("Test files not found")

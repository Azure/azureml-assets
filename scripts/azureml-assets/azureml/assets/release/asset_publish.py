# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Python script to publish assets."""

import argparse
import re
import shutil
import sys
import os
import yaml
from tempfile import TemporaryDirectory
from typing import List
from pathlib import Path
from string import Template
from subprocess import PIPE, run, STDOUT
import azureml.assets as assets
from azureml.assets.release.model_publish_utils import ModelUtils, MLFlowModelUtils
import azureml.assets.util as util
from azureml.assets.util import logger


ASSET_ID_TEMPLATE = Template("azureml://registries/$registry_name/$asset_type/$asset_name/versions/$version")
TEST_YML = "tests.yml"


def test_files_location(dir: Path):
    """Find test files in the directory."""
    test_jobs = []

    for test in dir.iterdir():
        logger.print("processing test folder: " + test.name)
        with open(test / TEST_YML) as fp:
            data = yaml.load(fp, Loader=yaml.FullLoader)
            for test_group in data.values():
                for test_job in test_group["jobs"].values():
                    test_jobs.append((test / test_job["job"]).as_posix())
    return test_jobs


def test_files_preprocess(test_jobs, asset_ids: dict):
    """Preprocess test files to generate asset ids."""
    for test_job in test_jobs:
        logger.print(f"processing test job: {test_job}")
        with open(test_job) as fp:
            data = yaml.load(fp, Loader=yaml.FullLoader)
            for job_name, job in data["jobs"].items():
                asset_name = job["component"]
                logger.print(f"processing asset {asset_name}")
                if asset_name in asset_ids:
                    job["component"] = asset_ids.get(asset_name)
                    logger.print(f"for job {job_name}, the new asset id: {job['component']}")
            with open(test_job, "w") as file:
                yaml.dump(data, file, default_flow_style=False, sort_keys=False)


def update_spec_file(spec_file: Path, path: Path):
    """Update the yaml file after getting the model has been prepared."""
    with open(spec_file) as f:
        model_file = yaml.safe_load(f)
    model_file["path"] = path
    with open(spec_file, "w") as f:
        yaml.dump(model_file, f)


def model_prepare(model_config: assets.ModelConfig, spec_file: Path) -> object:
    """Prepare Model."""
    """
    Prepare the model. Download the model if required.
    Convert the models to specified publish type.

    Return: returns the local path to the model.
    """
    result = {"download_success": False, "model_dir": None}

    if model_config.path_local_path:
        update_spec_file(spec_file, os.path.abspath(model_config.path_local_path))
        return result

    model_dir = TemporaryDirectory()

    model_utils = ModelUtils(
        model_url=model_config.package_url,
        model_commit_hash=model_config.package_commit_hash,
        model_dir=model_dir,
    )

    validate_download = model_utils.download_model()

    if validate_download is True:
        update_spec_file(spec_file, model_dir.name)

    if model_config.type.value == "mlflow_model":
        mlflow_utils = MLFlowModelUtils(
            name=model_config.name,
            task_name=model_config.task_name,
            flavor=model_config.flavor.value,
            model_dir=model_dir,
        )
        mlflow_utils.covert_into_mlflow_model()

    result["download_success"] = validate_download
    result["model_dir"] = model_dir

    return result


def model_clean(model_dir: TemporaryDirectory):
    """Delete the Model Artifact after the model has been pushed to the registry."""
    if model_dir is not None:
        print("Deleting model files from disk")
        try:
            model_dir.cleanup()
        except Exception as e:
            logger.print("Failed to Delete model from disk. Encountered error: " + e)


def run_command(cmd: str, failure_list: List, debug_mode: bool = None):
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

    return results


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
    parser.add_argument("-r", "--registry-name", required=True, type=str, help="the registry name")
    parser.add_argument("-s", "--subscription-id", type=str, help="the subscription ID")
    parser.add_argument("-g", "--resource-group", type=str, help="the resource group name")
    parser.add_argument("-w", "--workspace", type=str, help="the workspace name")
    parser.add_argument(
        "-a",
        "--assets-directory",
        required=True,
        type=Path,
        help="the assets directory",
    )
    parser.add_argument("-t", "--tests-directory", type=Path, help="the tests directory")
    parser.add_argument("-v", "--version-suffix", type=str, help="the version suffix")
    parser.add_argument("-l", "--publish-list", type=Path, help="the path of the publish list file")
    parser.add_argument(
        "-d",
        "--debug",
        type=_str2bool,
        nargs="?",
        const=True,
        default=False,
        help="debug mode",
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
            publish_list = config.get("create", {})
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
            logger.print(f"Skipping registering asset {asset.name} because it is not in the publish list")
            continue
        logger.print(f"Registering {asset}")
        final_version = asset.version + "-" + passed_version if passed_version else asset.version
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
            cmd = [
                shutil.which("az"),
                "ml",
                asset.type.value,
                "create",
                "--subscription",
                subscription_id,
                "--file",
                str(asset.spec_with_path),
                "--registry-name",
                registry_name,
                "--version",
                final_version,
            ]
            if resource_group:
                cmd.extend(["--resource-group", resource_group])
            if workspace:
                cmd.extend(["--workspace", workspace])
            if debug_mode:
                cmd.append("--debug")
            print(cmd)

            # Run command
            results = run_command(cmd, failure_list, debug_mode)

        elif asset.type == assets.AssetType.MODEL:

            model_config = asset.extra_config_as_object()
            result = model_prepare(model_config, asset.spec_with_path)
            # Assemble command
            if result["download_success"] is True:
                cmd = [
                    shutil.which("az"),
                    "ml",
                    asset.type.value,
                    "create",
                    "--subscription",
                    subscription_id,
                    "--file",
                    str(asset.spec_with_path),
                    "--registry-name",
                    registry_name,
                    "--version",
                    final_version,
                    "--tags",
                    str(model_config.tags),
                ]
                if resource_group:
                    cmd.extend(["--resource-group", resource_group])
                if workspace:
                    cmd.extend(["--workspace", workspace])
                print(cmd)

                # Run command
                results = run_command(
                    cmd,
                    failure_list,
                    debug_mode,
                )
                # Delete Model Artifacts
                model_clean(result["model_dir"])

        else:
            logger.log_warning(f"unsupported asset type: {asset.type.value}")

    if len(failure_list) > 0:
        logger.log_warning(f"following assets failed to publish: {failure_list}")

    logger.print("locating test files")
    if tests_dir:
        test_jobs = test_files_location(tests_dir)

        logger.print("preprocessing test files")
        test_files_preprocess(test_jobs, asset_ids)
        logger.print("finished preprocessing test files")
    else:
        logger.print("Test files not found")

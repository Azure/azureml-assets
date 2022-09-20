# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Python script to publish assets."""
from subprocess import check_call
import argparse
from pathlib import Path
import yaml
import azureml.assets as assets
import azureml.assets.util as util
from azureml.assets.util import logger
from string import Template
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
                    test_jobs.append((test / test_job['job']).as_posix())
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
                yaml.dump(
                    data,
                    file,
                    default_flow_style=False,
                    sort_keys=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--registry-name", required=True, type=str, help="the registry name")
    parser.add_argument("-g", "--resource-group", required=True, type=str, help="the resource group name")
    parser.add_argument("-s", "--subscription-id", required=True, type=str, help="the subscription-id")
    parser.add_argument("-w", "--workspace", required=True, type=str, help="the workspace name")
    parser.add_argument("-a", "--assets-directory", required=True, type=Path, help="the assets directory")
    parser.add_argument("-t", "--tests-directory", required=True, type=Path, help="the tests directory")
    parser.add_argument("--version-suffix", required=False, type=str, help="the version suffix")
    parser.add_argument("-l", "--publish-list", required=False, type=Path, help="the path of the publish list file")
    parser.add_argument("-d", "--debug", type=bool, default=False, help="debug mode")
    args = parser.parse_args()
    registry_name = args.registry_name
    subscription_id = args.subscription_id
    resource_group = args.resource_group
    workspace = args.workspace
    tests_dir = args.tests_directory
    component_dir = args.assets_directory
    passed_version = args.version_suffix
    publish_list_dir = args.publish_list
    debug_mode = args.debug
    publish_list = {}
    asset_ids = {}
    print("publishing assets")

    if publish_list_dir:
        with open(publish_list_dir) as fp:
            config = yaml.load(fp, Loader=yaml.FullLoader)
            publish_list = config.get('create', {})
            if not publish_list:
                logger.log_warning("The create list is empty.")
                exit(0)
            logger.print(f"create list: {publish_list}")

    assets_set = util.find_assets(
        input_dirs=component_dir,
        asset_config_filename=assets.DEFAULT_ASSET_FILENAME)
    
    failure_list = []
    for asset in assets_set:
        asset_names = publish_list.get(asset.type.value, [])
        if not ('*' in asset_names or asset.name in asset_names):
            logger.print(
                f"Skipping registering asset {asset.name} because it is not in the publish list")
            continue
        logger.print(f"Registering {asset.name}")
        final_version = asset.version
        spec_path = asset.spec_with_path
        if args.version_suffix:
            final_version = final_version + '-' + passed_version
        logger.print(f"final version: {final_version}")
        asset_ids[asset.name] = ASSET_ID_TEMPLATE.substitute(registry_name=registry_name,
                                                             asset_type=f"{asset.type.value}s",
                                                             asset_name=asset.name,
                                                             version=final_version)
        # switch case for asset type
        if asset.type == assets.AssetType.COMPONENT:
            cmd = f"az ml component create --subscription {subscription_id} " \
                f"--file {spec_path} --registry-name {registry_name} " \
                f"--version {final_version} --workspace {workspace} --resource-group {resource_group}"
            if debug_mode:
                cmd += " --debug> output.txt 2>&1"
            print(cmd)
            try:
                check_call(cmd, shell=True)
            except Exception as ex:
                logger.log_warning(
                    f"catch error creating {asset.type.value}: {asset.name} with exception {ex}")
                failure_list.append(asset.name)
            if debug_mode:
                check_call("cat output.txt | sed 's/Bearer.*$//'", shell=True)
        # TO-DO: add other asset types
        else:
            logger.log_warning(f"unsupported asset type: {asset.type.value}")
    if len(failure_list) > 0:
        logger.log_warning(f"following assets failed to publish: {failure_list}")

    logger.print('starting locating test files')
    test_jobs = test_files_location(tests_dir)
    logger.print('starting preprocessing test files')
    test_files_preprocess(test_jobs, asset_ids)
    logger.print('finished preprocessing test files')

# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""Python script to publish assets."""
from subprocess import check_call
import argparse
from pathlib import Path
import yaml
import azureml.assets as assets
import azureml.assets.util as util
from string import Template
ASSET_ID_TEMPLATE = Template(
    "azureml://registries/$registry_name/$asset_type/$asset_name/versions/$version")
TEST_YML = "tests.yml"
PROD_REGISTRY_NAME = "azureml"


def test_files_location(dir: Path):
    """Find test files in the directory."""
    test_jobs = []
    for test in dir.iterdir():
        print("processing test folder: " + test.name)
        with open(test / TEST_YML) as fp:
            data = yaml.load(fp, Loader=yaml.FullLoader)
            for test_group in data.values():
                for test_job in test_group['jobs'].values():
                    test_jobs.append((test / test_job['job']).as_posix())
    return test_jobs


def test_files_preprocess(test_jobs, asset_ids: dict):
    """Preprocess test files to generate asset ids."""
    for test_job in test_jobs:
        print(f"processing test job: {test_job}")
        with open(test_job) as fp:
            data = yaml.load(fp, Loader=yaml.FullLoader)
            for job_name, job in data["jobs"].items():
                asset_name = job["component"]
                print(f"processing asset {asset_name}")
                if asset_name in asset_ids:
                    job["component"] = asset_ids.get(asset_name)
                    print(f"for job {job_name}, the new asset id: {job['component']}")
            with open(test_job, "w") as file:
                yaml.dump(
                    data,
                    file,
                    default_flow_style=False,
                    sort_keys=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--registry-name",
        required=True,
        type=str,
        help="the registry name")
    parser.add_argument(
        "-g",
        "--resource-group",
        required=True,
        type=str,
        help="the resource group name")
    parser.add_argument(
        "-w",
        "--workspace",
        required=True,
        type=str,
        help="the workspace name")
    parser.add_argument(
        "-c",
        "--component-directory",
        required=True,
        type=Path,
        help="the component directory")
    parser.add_argument(
        "-t",
        "--tests-directory",
        required=True,
        type=Path,
        help="the tests directory")
    parser.add_argument(
        "-v",
        "--version",
        required=False,
        type=str,
        help="the version")
    parser.add_argument(
        "-l",
        "--whitelist",
        required=False,
        type=Path,
        help="the path of the whitelist file")
    args = parser.parse_args()
    registry_name = args.registry_name
    resource_group = args.resource_group
    workspace = args.workspace
    tests_dir = args.tests_directory
    component_dir = args.component_directory
    passed_version = args.version
    whitelist_dir = args.whitelist
    whitelist = []
    asset_ids = {}
    print("publishing assets")
    if registry_name == PROD_REGISTRY_NAME and not args.whitelist:
        print("No whitelist file provided for production registry")
        exit(1)
    if registry_name == PROD_REGISTRY_NAME:
        with open(whitelist_dir) as fp:
            data = yaml.load(fp, Loader=yaml.FullLoader)
            whitelist = [a for a in data]

    asset_version_with_build_id = registry_name + "." + passed_version
    print(f"generated componentVersionWithBuildId: {asset_version_with_build_id}")
    assets_set = util.find_assets(
        input_dirs=component_dir,
        asset_config_filename=assets.DEFAULT_ASSET_FILENAME)
    for asset in assets_set:
        if registry_name == PROD_REGISTRY_NAME and asset.name not in whitelist:
            print(
                f"Skipping registering asset {asset.name} because it is not in the whitelist")
            continue
        print(f"Registering {asset.name}")
        final_version = asset.version
        spec_path = asset.spec_with_path
        if registry_name != PROD_REGISTRY_NAME:
            final_version = final_version + '-' + asset_version_with_build_id
        print(f"final version: {final_version}")
        asset_ids[asset.name] = ASSET_ID_TEMPLATE.substitute(registry_name=registry_name,
                                                             asset_type=f"{asset.type.value}s",
                                                             asset_name=asset.name,
                                                             version=final_version)
        cmd = f"az ml component create --file {spec_path} --registry-name {registry_name} " \
                f"--version {final_version} --workspace {workspace} --resource-group {resource_group}"
        print(cmd)
        try:
            check_call(cmd, shell=True)
        except Exception as ex:
            print(
                f"catch error creating {asset.type}: {asset.name} with exception {ex}")
    print('All assets published')

    print('starting locating test files')
    test_jobs = test_files_location(tests_dir)
    print('starting preprocessing test files')
    test_files_preprocess(test_jobs, asset_ids)
    print('finished preprocessing test files')

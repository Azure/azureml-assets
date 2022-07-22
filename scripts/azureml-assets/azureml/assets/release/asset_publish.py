# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
from subprocess import check_call
import argparse
from pathlib import Path
import yaml
import azureml.assets as assets
import azureml.assets.util as util
from string import Template
ASSET_ID_TEMPLATE = Template(
    "azureml://registries/$registries_name/$asset_type/$asset_name/versions/$version")
TEST_YML = "tests.yml"
PROD_REGISTRY_NAME = "azureml"


def test_files_location(dir: Path):
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
    for test_job in test_jobs:
        print(f"processing test job: {test_job}")
        with open(test_job) as fp:
            data = yaml.load(fp, Loader=yaml.FullLoader)
            for job in data["jobs"]:
                asset_name = data["jobs"][job]["component"]
                print(f"processing asset {asset_name}")
                if asset_name in asset_ids:
                    data["jobs"][job]["component"] = asset_ids.get(asset_name)
                    print(f"New Asset ID: {data['jobs'][job]['component']}")
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
        help="the path of the whitelist file")
    args = parser.parse_args()
    registry_name = args.registry_name
    resource_group = args.resource_group
    workspace = args.workspace
    tests_dir = args.tests_directory
    component_dir = args.component_directory
    passed_version = args.version
    whitelist_dir = args.whitelist
    whitelist = None
    asset_ids = {}
    print("publishing assets")

    if registry_name == PROD_REGISTRY_NAME:
        whitelist = []
        with open(whitelist_dir) as fp:
            data = yaml.load(fp, Loader=yaml.FullLoader)
            for asset in data:
                whitelist.append(asset)

    asset_version_with_buildId = registry_name + "." + passed_version
    print("generated componentVersionWithBuildId: " + asset_version_with_buildId)
    asset_set = util.find_assets(
        input_dirs=component_dir,
        asset_config_filename=assets.DEFAULT_ASSET_FILENAME)
    for asset in asset_set:
        if registry_name == PROD_REGISTRY_NAME and asset.name not in whitelist:
            print(
                f"Skipping registering asset {asset.name} because it is not in the whitelist")
            continue
        else:
            print(f"Registering {asset.name}")
            final_version = asset.version
            spec_path = asset.spec_with_path
            if registry_name != PROD_REGISTRY_NAME:
                final_version = final_version + '-' + asset_version_with_buildId
            print(f"final version: {final_version}")
            asset_ids[asset.name] = ASSET_ID_TEMPLATE.substitute(registries_name=registry_name,
                                                                 asset_type=f"{asset.type.value}s",
                                                                 asset_name=asset.name,
                                                                 version=final_version)
            cmd_temp = Template(
                "az ml component create --file $f --registry-name $r --version $v --workspace $w --resource-group $g")
            cmd = cmd_temp.substitute(
                f=spec_path,
                r=registry_name,
                v=final_version,
                w=workspace,
                g=resource_group)
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

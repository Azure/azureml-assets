# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
from asyncio import subprocess
from subprocess import check_call
import argparse
from pathlib import Path
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("-r", "--registry-name", required=True, type=str, help="the registry name")
parser.add_argument("-g", "--resource-group", required=True, type=str, help="the resource group name")
parser.add_argument("-w", "--workspace", required=True, type=str, help="the workspace name")
parser.add_argument("-c", "--component-directory", required=True, type=Path, help="the component directory")
parser.add_argument("-t", "--tests-directory", required=True, type=Path, help="the tests directory")
parser.add_argument("-v", "--version", required=False, type=str, help="the version")
args = parser.parse_args()
registry_name = args.registry_name
resource_group = args.resource_group
workspace = args.workspace
tests_dir = args.tests_directory
component_dir = args.component_directory
passed_version = args.version
subprocess


def test_files_location(dir: Path):
    test_jobs = []
    for x in dir.iterdir():
        print("processing test folder: " + x.name)
        with open(x / 'tests.yml') as fp:
            data = yaml.load(fp, Loader=yaml.FullLoader)
            for test_group in data.values():
                for test_job in test_group['jobs'].values():
                    test_jobs.append((x / test_job['job']).as_posix())
    return test_jobs


def process_asset_id(asset_id, full_version):
    list = asset_id.split("/")
    if len(full_version) > 0:
        list[-1] += '-'+full_version
    list[-5] = registry_name
    return "/".join(list)


def test_files_preprocess(test_jobs, full_version):
    for test_job in test_jobs:
        print(f"processing test job: {test_job}")
        with open(test_job) as fp:
            data = yaml.load(fp, Loader=yaml.FullLoader)
            for job in data["jobs"]:
                original_asset = data["jobs"][job]["component"]
                print(f"processing asset {original_asset}")
                if original_asset.startswith("azureml"):
                    new_asset = process_asset_id(original_asset, full_version)
                    data["jobs"][job]["component"] = new_asset
                    print(data["jobs"][job]["component"])
            with open(test_job, "w") as file:
                yaml.dump(data, file, default_flow_style=False)


print("publishing assets")

componentVersionWithBuildId = ""
if registry_name != "azureml":
    componentVersionWithBuildId = registry_name + "." + passed_version
print("generated componentVersionWithBuildId: " + componentVersionWithBuildId)
print('starting locating test files')
test_jobs = test_files_location(tests_dir)
print('starting preprocessing test files')
test_files_preprocess(test_jobs, componentVersionWithBuildId)
print('finished preprocessing test files')
for x in component_dir.iterdir():
    if x.name != "src":
        print("Registering " + x.name)
        with open(x / 'asset.yaml') as fp:
            data = yaml.load(fp, Loader=yaml.FullLoader)
        spec_file = data['spec']
        spec_path = x / spec_file
        print("Does spec path exist: "+spec_path.is_file().__str__())
        final_version = ''
        with open(spec_path) as fp:
            spec_data = data = yaml.load(fp, Loader=yaml.FullLoader)
            if registry_name != "azureml":
                final_version = spec_data['version'] + '-' + componentVersionWithBuildId
            print("final version: "+final_version)
        print(f"az ml component create --file {spec_path} --registry-name {registry_name} --version {final_version} --workspace {workspace} --resource-group {resource_group} ")
        try:
            check_call(f"az ml component create --file {spec_path} --registry-name {registry_name} --version {final_version} --workspace {workspace}  --resource-group {resource_group}", shell=True)
        except Exception as ex:
            print(f"catch error creating component {x} with exception {ex}")
print('All assets published')

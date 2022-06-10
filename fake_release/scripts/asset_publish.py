from asyncio import subprocess
import argparse
from pathlib import Path
import os
import sys
import datetime
import subprocess
import ruamel.yaml

parser = argparse.ArgumentParser()
parser.add_argument("-r", "--registry-name", required=True, type=str, help="the registry name")
parser.add_argument("-g", "--resource-group", required=True, type=str, help="the resource group name")
parser.add_argument("-w", "--workspace", required=True, type=str, help="the workspace name")
parser.add_argument("-c", "--component-directory", required=True, type=Path, help="the component directory")
parser.add_argument("-t", "--tests-directory", required=True, type=Path, help="the tests directory")
args = parser.parse_args()
registry_name = args.registry_name
resource_group = args.resource_group
workspace = args.workspace
component_dir = args.component_directory
tests_dir = args.tests_directory

def test_files_location(DIR: Path):
    test_jobs = []
    yaml = ruamel.yaml.YAML()
    for x in os.listdir(DIR.__str__()):
        print("processing test folder: " + x)
        area_folder = DIR.__str__() + "/" + x
        with open(area_folder+'/tests.yml') as fp:
            data = yaml.load(fp)
            for test_group in data:
                for test_job in data[test_group]['jobs']:
                    test_jobs.append(area_folder + '/' + data[test_group]['jobs'][test_job]['job'])
    return test_jobs

def process_asset_id(asset_id, full_version):
    list = asset_id.split("/")
    list[-1] += full_version
    list[-5] = registry_name
    return "/".join(list)

def test_files_preprocess(test_jobs, full_version):
    yaml = ruamel.yaml.YAML()
    for test_job in test_jobs:
        print("processing test job: " + test_job)
        with open(test_job) as fp:
            data = yaml.load(fp)
            for job in data["jobs"]:
                print("processing asset"+data["jobs"][job]["component"])
                original_asset = data["jobs"][job]["component"]
                new_asset = process_asset_id(original_asset, full_version)
                data["jobs"][job]["component"] = new_asset
                print(data["jobs"][job]["component"])
            with open(test_job, "w") as file:
                yaml.dump(data, file)



print("publishing assets")

timestamp = datetime.datetime.now()
subprocess.call('az extension add --source https://azuremlsdktestpypi.blob.core.windows.net/wheels/azureml-v2-cli-e2e-test/62707480/ml-0.0.62707480-py3-none-any.whl --pip-extra-index-urls https://azuremlsdktestpypi.azureedge.net/azureml-v2-cli-e2e-test/62707480 --yes')
subprocess.call('export AZURE_ML_CLI_PRIVATE_FEATURES_ENABLED=true')
subprocess.call('az ml -h')

componentVersionWithBuildId="ev2."+registry_name+"."+timestamp
print('starting locating test files')
test_jobs = test_files_location(tests_dir)
print('starting preprocessing test files')
test_files_preprocess(test_jobs, componentVersionWithBuildId)
print('finished preprocessing test files')
yaml = ruamel.yaml.YAML()
for x in os.listdir(component_dir.__str__()):
    print("Registering "+x)
    with open(component_dir.__str__()+"/"+x) as fp:
        data = yaml.load(fp)
    spec_file = data['spec']
    print(f"az ml component create --file {x}/{spec_file} --registry {registry_name} --version {componentVersionWithBuildId} --workspace {workspace}  --resource-group {resource_group} --set environment='azureml://registries/CuratedRegistry/environments/AzureML-minimal-ubuntu18.04-py37-cpu-inference/versions/34' ")
    subprocess.check_call(f"az ml component create --file {x}/{spec_file} --registry {registry_name} --version {componentVersionWithBuildId} --workspace {workspace}  --resource-group {resource_group} --set environment='azureml://registries/CuratedRegistry/environments/AzureML-minimal-ubuntu18.04-py37-cpu-inference/versions/34' --debug")
print('All assets published')
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

def test_files_preprocess(DIR: Path, full_version: str):
    yaml = ruamel.yaml.YAML()
    for x in os.listdir(DIR.__str__()):
        print("processing test file: " + x)
        with open(x) as fp:
            data = yaml.load(fp)
        for job in data["jobs"]:
            print("processing asset"+data["jobs"][job]["component"])
            original_asset = data["jobs"][job]["component"]
            new_asset = process_asset_id(original_asset, full_version)
            data["jobs"][job]["component"] = new_asset
            print(data["jobs"][job]["component"])
        with open(DIR.__str__()+"/"+x, "w") as file:
            yaml.dump(data, file)

def process_asset_id(asset_id, full_version):
    list = asset_id.split("/")
    list[-1] += full_version
    list[-5] = registry_name
    return "/".join(list)

print("publishing assets")

timestamp = datetime.datetime.now()
subprocess.call('az extension add --source https://azuremlsdktestpypi.blob.core.windows.net/wheels/azureml-v2-cli-e2e-test/62707480/ml-0.0.62707480-py3-none-any.whl --pip-extra-index-urls https://azuremlsdktestpypi.azureedge.net/azureml-v2-cli-e2e-test/62707480 --yes')
subprocess.call('export AZURE_ML_CLI_PRIVATE_FEATURES_ENABLED=true')
subprocess.call('az ml -h')

componentVersionWithBuildId="ev2."+registry_name+"."+timestamp
test_files_preprocess(tests_dir, componentVersionWithBuildId)
yaml = ruamel.yaml.YAML()
for x in os.listdir(component_dir.__str__()):
    print("Registering "+x)
    with open(component_dir.__str__()+"/"+x) as fp:
        data = yaml.load(fp)
    spec_file = data['spec']
    print(f"az ml component create --file {x}/{spec_file} --registry {registry_name} --version {componentVersionWithBuildId} --workspace {workspace}  --resource-group {resource_group} --set environment='azureml://registries/CuratedRegistry/environments/AzureML-minimal-ubuntu18.04-py37-cpu-inference/versions/34' ")
    subprocess.check_call(f"az ml component create --file {x}/{spec_file} --registry {registry_name} --version {componentVersionWithBuildId} --workspace {workspace}  --resource-group {resource_group} --set environment='azureml://registries/CuratedRegistry/environments/AzureML-minimal-ubuntu18.04-py37-cpu-inference/versions/34' --debug")
print('All assets published')
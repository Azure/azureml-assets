from asyncio import subprocess
from pathlib import Path
import os
import sys
import datetime
import subprocess

REGISTRY_NAME = sys.argv[1]
WORKSPACE_NAME = sys.argv[2]
RESOURCE_GROUP_NAME = sys.argv[3]

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
    list[-5] = REGISTRY_NAME
    return "/".join(list)

print("publishing assets")

timestamp = datetime.datetime.now()
subprocess.call('az extension add --source https://azuremlsdktestpypi.blob.core.windows.net/wheels/azureml-v2-cli-e2e-test/62707480/ml-0.0.62707480-py3-none-any.whl --pip-extra-index-urls https://azuremlsdktestpypi.azureedge.net/azureml-v2-cli-e2e-test/62707480 --yes')
subprocess.call('export AZURE_ML_CLI_PRIVATE_FEATURES_ENABLED=true')
subprocess.call('az ml -h')
component_dir = Path("azureml-assets/latest/component")
test_dir = Path("azureml-assets/latest/test")
componentVersionWithBuildId="ev2."+REGISTRY_NAME+"."+timestamp
test_files_preprocess(test_dir, componentVersionWithBuildId)
for x in os.listdir(component_dir.__str__()):
    print("Registering "+x)
    print(f"az ml component create --file {x}/spec.yaml --registry {REGISTRY_NAME} --version {componentVersionWithBuildId} --workspace {WORKSPACE_NAME}  --resource-group {RESOURCE_GROUP_NAME} --set environment='azureml://registries/CuratedRegistry/environments/AzureML-minimal-ubuntu18.04-py37-cpu-inference/versions/34' ")
    subprocess.call(f"az ml component create --file {x}/spec.yaml --registry {REGISTRY_NAME} --version {componentVersionWithBuildId} --workspace {WORKSPACE_NAME}  --resource-group {RESOURCE_GROUP_NAME} --set environment='azureml://registries/CuratedRegistry/environments/AzureML-minimal-ubuntu18.04-py37-cpu-inference/versions/34' --debug")
print('All assets published')
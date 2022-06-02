from asyncio import subprocess
from pathlib import Path
import os
import sys
import datetime
import subprocess

REGISTRY_NAME = sys.argv[1]
WORKSPACE_NAME = sys.argv[2]
RESOURCE_GROUP_NAME = sys.argv[3]

print("asset_publish")

timestamp = datetime.datetime.now()
subprocess.call('az extension add --source https://azuremlsdktestpypi.blob.core.windows.net/wheels/azureml-v2-cli-e2e-test/62707480/ml-0.0.62707480-py3-none-any.whl --pip-extra-index-urls https://azuremlsdktestpypi.azureedge.net/azureml-v2-cli-e2e-test/62707480 --yes')
subprocess.call('export AZURE_ML_CLI_PRIVATE_FEATURES_ENABLED=true')
subprocess.call('az ml -h')
DIR = Path("azureml-assets/latest/component")
componentVersionWithBuildId="ev2."+REGISTRY_NAME+"."+timestamp
for x in os.listdir(DIR.__str__()):
    print("Registering "+x)
    print(f"az ml component create --file {x}/spec.yaml --registry {REGISTRY_NAME} --version {componentVersionWithBuildId} --workspace {WORKSPACE_NAME}  --resource-group {RESOURCE_GROUP_NAME} --set environment='azureml://registries/CuratedRegistry/environments/AzureML-minimal-ubuntu18.04-py37-cpu-inference/versions/34' ")
    subprocess.call(f"az ml component create --file {x}/spec.yaml --registry {REGISTRY_NAME} --version {componentVersionWithBuildId} --workspace {WORKSPACE_NAME}  --resource-group {RESOURCE_GROUP_NAME} --set environment='azureml://registries/CuratedRegistry/environments/AzureML-minimal-ubuntu18.04-py37-cpu-inference/versions/34' --debug")
print('All assets published')
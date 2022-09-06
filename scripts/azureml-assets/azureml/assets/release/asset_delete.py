# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Python script to publish assets."""
import argparse
from pathlib import Path
import yaml
import requests
import json
from string import Template
import azureml.assets as assets
from azureml.assets.util import logger
from collections import defaultdict

ASSET_ID_TEMPLATE = Template(
    "azureml://registries/$registry_name/$asset_type/$asset_name/versions/$version")

CONTAINER_ASSET_ID_TEMPLATE = Template(
    "azureml://registries/$registry_name/$asset_type/$asset_name")

VERSION_CHECK_TEMPLATE = Template("$base_url/mferp/managementfrontend/subscriptions/$subscription/resourceGroups/$resource_group/providers/Microsoft.MachineLearningServices/registries/$registry/$asset_type/$asset_name/versions?api-version=2021-10-01-dataplanepreview")
    
def construct_asset_id(asset_name: str, asset_version: str, asset_type: str, registry_name: str):
    return ASSET_ID_TEMPLATE.substitute(
        registry_name=registry_name,
        asset_type=f'{asset_type}s',
        asset_name=asset_name,
        version=asset_version)

def delete_single_version(base_url: str, asset_id: str, token: str):
    target_url = f'{base_url}/assetstore/v1.0/delete'
    payload=json.dumps({"id": asset_id }) # "azureml://registries/azureml-dev/components/pytorch_image_classifier/versions/1.0.4"
    headers={'Authorization': f'Bearer {token}',  'Content-Type': 'application/json'}
    response=requests.request("DELETE", target_url, headers=headers, data=payload)
    if response.status_code != 202:
        logger.log_warning(f"Deletion failed for {asset_id}")


def delete_all_versions(base_url: str, asset_name: str, asset_type: str, token: str, subscription: str, resource_group: str, registry: str):
    version_list = list_all_version(base_url, asset_name, asset_type, token, subscription, resource_group, registry)
    for version in version_list[1:]:
        asset_id = construct_asset_id(asset_name, version, asset_type, registry)
        delete_single_version(base_url, asset_id, token)
    default_version = version_list[0]
    asset_id = construct_asset_id(asset_name, default_version, asset_type, registry)
    delete_single_version(base_url, asset_id, token)
    container_id = CONTAINER_ASSET_ID_TEMPLATE.substitute(asset_name=asset_name, asset_type=f'{asset_type}s', registry_name=registry)
    delete_single_version(base_url, container_id, token)

def is_latest_version(base_url: str, asset_name: str, asset_version: str, asset_type: str, token: str, subscription: str, resource_group: str, registry: str):
    version_list = list_all_version(base_url, asset_name, asset_type, token, subscription, resource_group, registry)
    if len(version_list) == 0:
        logger.log_warning(f"No versions are found for published {asset_type}: {asset_name}. Please check.")
    elif asset_version == version_list[0]:
        logger.log_warning(f"{asset_type}: {asset_name} with version {asset_version} cannot be deleted because it is the default version")
    return len(version_list)>0 and asset_version == version_list[0]

def list_all_version(base_url: str, asset_name: str, asset_type: str, token: str, subscription: str, resource_group: str, registry: str):
    target_url = VERSION_CHECK_TEMPLATE.substitute(base_url=base_url, subscription=subscription, resource_group=resource_group, registry=registry, asset_type=f'{asset_type}s', asset_name=asset_name)
    headers={'Authorization': f'Bearer {token}'}
    response=requests.request("GET", target_url, headers=headers)
    response_detail = response.json()
    version_list = []
    for version in response_detail['value']:
        version_list.append(version['name'])
    return version_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--deletion-list", required=True, type=Path, help="the path of deletion list file")
    parser.add_argument("--url", required=True, type=str, help="the base url of the api")
    parser.add_argument("-r", "--registry-name", required=True, type=str, help="the registry name")
    parser.add_argument("-g", "--resource-group", required=True, type=str, help="the resource group name")
    parser.add_argument("-s", "--subscription-id", required=True, type=str, help="the subscription-id")
    parser.add_argument("-t", "--token", required=True, type=str, help="the Bearer token")
    args = parser.parse_args()
    subscription_id = args.subscription_id
    resource_group = args.resource_group
    registry_name = args.registry_name
    token = args.token
    url = args.url
    deletion_list = defaultdict(list)
    with open(args.deletion_list) as fp:
        deletion_yaml = yaml.load(fp, Loader=yaml.FullLoader)
        for asset_type, asset_list in deletion_yaml.items():
             for asset in asset_list:
                deletion_list[asset_type].append({'name': asset['name'], 'version': asset['version']})
    print(f'deletion_list: {deletion_list}')

    for asset_type in deletion_list:
        if asset_type == assets.AssetType.COMPONENT.value:
            for asset in deletion_list[asset_type]:
                if asset['version'] == '*':
                    delete_all_versions(url, asset['name'], asset_type, token, subscription_id, resource_group, registry_name)
                else:
                    if is_latest_version(url, asset['name'], asset['version'], asset_type, token, subscription_id, resource_group, registry_name):
                        continue
                    asset_id = construct_asset_id(asset['name'], asset['version'], asset_type, registry_name)
                    print(f'delete asset: {asset_id}')
                    delete_single_version(url, asset_id, token)
        
        # TO-DO: add other asset types
        else:
            print(f"Deletion failed: unsupported asset type: {asset_type}")


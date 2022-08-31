# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Python script to publish assets."""
from subprocess import check_call
import argparse
from pathlib import Path
import yaml
import requests
import json
from string import Template
import azureml.assets as assets
from collections import defaultdict

ASSET_ID_TEMPLATE = Template(
    "azureml://registries/$registry_name/$asset_type/$asset_name/versions/$version")

def delete_single_version(url: str, asset_id: str, token: str):
    payload=json.dumps({"id": asset_id }) # "azureml://registries/azureml-dev/components/pytorch_image_classifier/versions/1.0.4"
    headers={'Authorization': f'Bearer {token}',  'Content-Type': 'application/json'}
    response=requests.request("DELETE", url, headers=headers, data=payload)
    print(response.text)


def delete_all_versions(url: str, asset_id: str, token: str):


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--deletion-list", required=True, type=Path, help="the path of deletion list file")
    parser.add_argument("-u, --url", required=True, type=str, help="the url of the deletion api")
    parser.add_argument("-r", "--registry-name", required=True, type=str, help="the registry name")
    parser.add_argument("-g", "--resource-group", required=True, type=str, help="the resource group name")
    parser.add_argument("-s", "--subscription-id", required=True, type=str, help="the subscription-id")
    parser.add_argument("-w", "--workspace", required=True, type=str, help="the workspace name")
    parser.add_argument("-t", "--token", required=True, type=str, help="the Bearer token")
    args = parser.parse_args()
    subscription_id = args.subscription_id
    resource_group = args.resource_group
    workspace = args.workspace
    registry_name = args.registry_name
    token = args.token
    url = args.url
    deletion_list = defaultdict(list)
    with open(args.deletion_list) as fp:
        deletion_yaml = yaml.load(fp, Loader=yaml.FullLoader)
        for asset_type, asset_list in deletion_yaml.items():
             for asset in asset_list:
                asset_name, asset_version = asset['name'], asset['version']
                asset_id = ASSET_ID_TEMPLATE.substitute(registry_name=registry_name, asset_type=f'{asset_type}s', asset_name=asset_name, version=asset_version)
                deletion_list[asset_type].append(asset_id)

    for asset_type in deletion_list:
        if asset_type == assets.AssetType.COMPONENT.value:
            for asset_id in deletion_list[asset_type]:
                if asset_id.endswith('*'):
                    delete_all_versions(url, asset_id, token)
                else:
                    delete_single_version(url, asset_id, token)
        
        # TO-DO: add other asset types
        else:
            print(f"Deletion failed: unsupported asset type: {asset_type}")


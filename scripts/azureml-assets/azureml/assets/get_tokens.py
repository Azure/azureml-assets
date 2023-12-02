# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Generate SAS tokens for models."""

import argparse
import re
from pathlib import Path
from typing import List

import azureml.assets as assets
import azureml.assets.util as util
from azureml.assets.config import AssetType, AzureBlobstoreAssetPath

from collections import defaultdict
from datetime import timedelta
import json


def get_tokens(input_dirs: List[Path],
               asset_config_filename: str,
               json_output_path: str,
               pattern: re.Pattern = None):
    """Generate SAS tokens for models to JSON output file.

    Args:
        input_dirs (List[Path]): List of directories to search for assets.
        asset_config_filename (str): Asset config filename to search for.
        json_output_path (str): Path of JSON file to write output to.
        pattern (re.Pattern, optional): Regex pattern for assets to copy. Defaults to None.
    """
    json_info = defaultdict(dict)
    
    # placeholder test
    print('In updated get_tokens')

    # Check prompt asssets
    for asset_config in util.find_assets(
            input_dirs, asset_config_filename, types=[AssetType.PROMPT], pattern=pattern):
        print('found asset', asset_config)
        prompt_config: assets.GenericAssetConfig = asset_config.extra_config_as_object()
        path = prompt_config.path
        print('found path:', path)
        account_name = prompt_config.path.storage_name
        container_name = prompt_config.path.container_name
        print('found account', account_name, 'container', container_name)
        _ = path.get_uri(token_expiration=timedelta(days=1))
        token = path.token
        json_info[account_name][container_name] = token

    # Filter to only models
    for asset_config in util.find_assets(
            input_dirs, asset_config_filename, types=[AssetType.MODEL], pattern=pattern):

        model_config: assets.ModelConfig = asset_config.extra_config_as_object()
        path = model_config.path

        if not isinstance(path, AzureBlobstoreAssetPath):
            continue

        account_name = model_config.path.storage_name
        container_name = model_config.path.container_name

        if container_name in json_info[account_name]:
            continue

        _ = model_config.path.get_uri(token_expiration=timedelta(days=1))
        token = model_config.path.token

        if token is not None and len(token) == 0:
            token = None

        json_info[account_name][container_name] = token

    with open(json_output_path, 'w') as json_token_file:
        json.dump(json_info, json_token_file)


if __name__ == '__main__':
    # Handle command-line args
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-dirs", required=True,
                        help="Comma-separated list of directories containing assets")
    parser.add_argument("-a", "--asset-config-filename", default=assets.DEFAULT_ASSET_FILENAME,
                        help="Asset config file name to search for")
    parser.add_argument("-j", "--json-output-path", required=True,
                        help="Path of JSON file to output to")
    parser.add_argument("-t", "--pattern", type=re.compile,
                        help="Regex pattern to select assets to copy, in the format <type>/<name>/<version>")
    args = parser.parse_args()

    # Convert comma-separated values to lists
    input_dirs = [Path(d) for d in args.input_dirs.split(",")]

    # Get SAS tokens
    get_tokens(input_dirs=input_dirs,
               asset_config_filename=args.asset_config_filename,
               json_output_path=args.json_output_path,
               pattern=args.pattern)

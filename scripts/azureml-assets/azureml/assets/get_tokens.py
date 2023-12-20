# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Generate SAS tokens for models."""

import argparse
import re
from pathlib import Path
from typing import List

import azureml.assets as assets
import azureml.assets.util as util
from azureml.assets.config import AssetType, AzureBlobstoreAssetPath, GENERIC_ASSET_TYPES

from collections import defaultdict
from datetime import timedelta
import json


DEFAULT_SAS_EXPIRATION_TIMEOUT = timedelta(days=3)


def get_tokens(input_dirs: List[Path],
               asset_config_filename: str,
               json_output_path: str,
               pattern: re.Pattern = None):
    """Generate SAS tokens for models and generic assets to JSON output file.

    Args:
        input_dirs (List[Path]): List of directories to search for assets.
        asset_config_filename (str): Asset config filename to search for.
        json_output_path (str): Path of JSON file to write output to.
        pattern (re.Pattern, optional): Regex pattern for assets to copy. Defaults to None.
    """
    json_info = defaultdict(dict)

    # Generate SAS tokens for generic assets
    asset_types = [AssetType.MODEL] + GENERIC_ASSET_TYPES
    for asset_config in util.find_assets(
            input_dirs, asset_config_filename, types=asset_types, pattern=pattern):

        if asset_config.type == AssetType.MODEL:
            model_config: assets.ModelConfig = asset_config.extra_config_as_object()
            if isinstance(model_config.path, AzureBlobstoreAssetPath):
                add_token_info(model_config.path, json_info)

        elif asset_config.type in GENERIC_ASSET_TYPES:
            generic_config: assets.GenericAssetConfig = asset_config.extra_config_as_object()
            if generic_config and isinstance(generic_config.path, AzureBlobstoreAssetPath):
                add_token_info(generic_config.path, json_info)

    with open(json_output_path, 'w') as json_token_file:
        json.dump(json_info, json_token_file)


def add_token_info(path: AzureBlobstoreAssetPath, json_info: defaultdict(dict)):
    """Generate a SAS token and add it to the json info token dictionary.

    Args:
        storage_path (AzureBlobstoreAssetPath): Blob storage path to update.
        json_info (defaultdict(dict)): Dictionary used to generate the JSON token file.
    """
    account_name = path.storage_name
    container_name = path.container_name

    if container_name in json_info[account_name]:
        return

    _ = path.get_uri(token_expiration=DEFAULT_SAS_EXPIRATION_TIMEOUT)
    token = path.token

    if token is not None and len(token) == 0:
        token = None

    json_info[account_name][container_name] = token


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

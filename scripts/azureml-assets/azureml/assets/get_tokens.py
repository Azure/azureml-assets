# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Generate SAS tokens for models."""

import re
import json
import argparse

from pathlib import Path
from typing import List
from collections import defaultdict
from datetime import timedelta

import azureml.assets as assets
import azureml.assets.util as util
from azureml.assets.config import AssetType, AzureBlobstoreAssetPath, GENERIC_ASSET_TYPES


DEFAULT_SAS_EXPIRATION_HRS = 72  # 3 days
MAX_SAS_EXPIRATION_HRS = 168  # 7 days


def get_tokens(input_dirs: List[Path],
               asset_config_filename: str,
               json_output_path: str,
               sas_expiration_hrs: int,
               pattern: re.Pattern = None):
    """Generate SAS tokens for models, datasets, and generic assets to JSON output file.

    Args:
        input_dirs (List[Path]): List of directories to search for assets.
        asset_config_filename (str): Asset config filename to search for.
        sas_expiration_hrs (int): Storage SAS expiration in hrs
        json_output_path (str): Path of JSON file to write output to.
        pattern (re.Pattern, optional): Regex pattern for assets to copy. Defaults to None.
    """
    json_info = defaultdict(dict)

    # Filter to only models, datasets, and generic assets
    asset_types = [AssetType.MODEL, AssetType.DATA] + GENERIC_ASSET_TYPES
    for asset_config in util.find_assets(
            input_dirs, asset_config_filename, types=asset_types, pattern=pattern):

        if asset_config.type == AssetType.MODEL:
            model_config: assets.ModelConfig = asset_config.extra_config_as_object()
            if model_config and isinstance(model_config.path, AzureBlobstoreAssetPath):
                add_token_info(model_config.path, json_info, sas_expiration_hrs)

        elif asset_config.type == AssetType.DATA:
            data_config: assets.DataConfig = asset_config.extra_config_as_object()
            if data_config and isinstance(data_config.path, AzureBlobstoreAssetPath):
                add_token_info(data_config.path, json_info, sas_expiration_hrs)

        elif asset_config.type in GENERIC_ASSET_TYPES:
            generic_config: assets.GenericAssetConfig = asset_config.extra_config_as_object()
            if generic_config and isinstance(generic_config.path, AzureBlobstoreAssetPath):
                add_token_info(generic_config.path, json_info, sas_expiration_hrs)

    with open(json_output_path, 'w') as json_token_file:
        json.dump(json_info, json_token_file)


def add_token_info(path: AzureBlobstoreAssetPath, json_info: defaultdict(dict), sas_expiration_hrs: int):
    """Generate a SAS token and add it to the json info token dictionary.

    Args:
        storage_path (AzureBlobstoreAssetPath): Blob storage path to update.
        json_info (defaultdict(dict)): Dictionary used to generate the JSON token file.
        sas_expiration_hrs (int): Storage SAS expiration in hrs
    """
    account_name = path.storage_name
    container_name = path.container_name

    if container_name in json_info[account_name]:
        return

    _ = path.get_uri(token_expiration=timedelta(hours=sas_expiration_hrs))
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
    parser.add_argument("-s", "--sas-expiration-hrs", type=int, required=False,
                        default=DEFAULT_SAS_EXPIRATION_HRS,
                        help=f"SAS expiration in hours. Default is {DEFAULT_SAS_EXPIRATION_HRS} hours.")
    parser.add_argument("-t", "--pattern", type=re.compile,
                        help="Regex pattern to select assets to copy, in the format <type>/<name>/<version>")
    args = parser.parse_args()

    # Convert comma-separated values to lists
    input_dirs = [Path(d) for d in args.input_dirs.split(",")]

    sas_expiration_hrs = args.sas_expiration_hrs
    # make sure token expiration is well bound under limits
    if sas_expiration_hrs > sas_expiration_hrs:
        sas_expiration_hrs = MAX_SAS_EXPIRATION_HRS
    elif sas_expiration_hrs <= 0:
        sas_expiration_hrs = DEFAULT_SAS_EXPIRATION_HRS

    # Get SAS tokens
    get_tokens(input_dirs=input_dirs,
               asset_config_filename=args.asset_config_filename,
               json_output_path=args.json_output_path,
               sas_expiration_hrs=sas_expiration_hrs,
               pattern=args.pattern)

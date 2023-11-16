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
import datetime
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

    # Filter to only models
    for asset_config in util.find_assets(
            input_dirs, asset_config_filename, types=[AssetType.MODEL], pattern=pattern):

        model_config: assets.ModelConfig = asset_config.extra_config_as_object()

        if not isinstance(model_config.path, AzureBlobstoreAssetPath):
            continue

        account_name = model_config.path.storage_name
        container_name = model_config.path.container_name

        if container_name in json_info[account_name]:
            continue

        start_time = datetime.datetime.now(datetime.timezone.utc)
        expiry_time = start_time + datetime.timedelta(days=1)

        token = AzureBlobstoreAssetPath.generate_sas_token(account_uri=model_config.path.account_uri, container_name=container_name, storage_name=account_name, start_time=start_time, expiry_time=expiry_time)

        if len(token) == 0:
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

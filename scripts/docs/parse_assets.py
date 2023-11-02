# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Parse assets from directory and generate markdown files."""

import argparse
import re
from pathlib import Path
from typing import List

import azureml.assets as assets
import azureml.assets.util as util

from generate_asset_documentation import AssetInfo, Categories


def parse_assets(input_dirs: List[Path],
                 asset_config_filename: str,
                 pattern: re.Pattern = None):
    """Parse all assets from input directory and generate documentation for each.

    Args:
        input_dirs (List[Path]): List of directories to search for assets.
        asset_config_filename (str): Asset config filename to search for.
        pattern (re.Pattern, optional): Regex pattern for assets to parse. Defaults to None.
    """
    categories = Categories()

    for asset_config in util.find_assets(input_dirs, asset_config_filename, pattern=pattern):
        asset_info = AssetInfo.create_asset_info(asset_config)
        if asset_info:
            categories.classify_asset(asset_info)
            # Save asset info. Revisit to save all docs after
            asset_info.save()

    categories.save()


if __name__ == '__main__':
    # Handle command-line args
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-dirs", required=True,
                        help="Comma-separated list of directories containing assets")
    parser.add_argument("-a", "--asset-config-filename", default=assets.DEFAULT_ASSET_FILENAME,
                        help="Asset config file name to search for")
    parser.add_argument("-t", "--pattern", type=re.compile,
                        help="Regex pattern to select assets to parse, in the format <type>/<name>/<version>")
    args = parser.parse_args()

    # Convert comma-separated values to lists
    input_dirs = [Path(d) for d in args.input_dirs.split(",")]

    # Parse assets
    parse_assets(input_dirs=input_dirs,
                 asset_config_filename=args.asset_config_filename,
                 pattern=args.pattern)

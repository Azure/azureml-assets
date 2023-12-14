# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Create a test matrix for GitHub workflow job."""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import List

import azureml.assets as assets
import azureml.assets.util as util
from azureml.assets.util import logger

TEST_COUNT = "test_count"
COUNTERS = [TEST_COUNT]
MATRIX = "matrix"


def create_test_matrix(input_dirs: List[Path],
                       asset_config_filename: str,
                       changed_files: List[Path]):
    """Create test matrix.

    Args:
        input_dirs (List[Path]): Directories to search for assets.
        asset_config_filename (str): Asset config filename to search input_dirs for.
        changed_files (List[Path]): List of changed files used to select only assets.
    """
    counters = Counter()
    asset_config_files = []
    asset_test_dirs = []
    for asset_config in util.find_assets(input_dirs, asset_config_filename, changed_files=changed_files):
        # Skip assets without testing enabled
        if not asset_config.pytest_enabled:
            logger.log_debug(f"Testing is not enabled for {asset_config}")
            continue

        # Ensure test directory hasn't already been added
        test_dir = asset_config.pytest_tests_dir_with_path
        if test_dir in asset_test_dirs:
            logger.log_debug(f"Skipping {asset_config} because {test_dir} has already been added")
            continue

        # Store asset and create matrix
        logger.log_debug(f"Adding {asset_config.file_path} with tests at {test_dir} to testing matrix")
        asset_config_files.append(str(asset_config.file_path))
        counters[TEST_COUNT] += 1

    # Set variables
    for counter_name in COUNTERS:
        logger.set_output(counter_name, counters[counter_name])
    logger.set_output(MATRIX, json.dumps({'asset_config_path': asset_config_files}))


if __name__ == '__main__':
    # Handle command-line args
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-dirs", required=True,
                        help="Comma-separated list of directories containing assets to test")
    parser.add_argument("-a", "--asset-config-filename", default=assets.DEFAULT_ASSET_FILENAME,
                        help="Asset config file name to search for")
    parser.add_argument("-c", "--changed-files", help="Comma-separated list of changed files, used to filter assets")
    args = parser.parse_args()

    # Convert comma-separated values to lists
    input_dirs = [Path(d) for d in args.input_dirs.split(",")]
    changed_files = [Path(f) for f in args.changed_files.split(",")] if args.changed_files else []

    # Test assets
    create_test_matrix(input_dirs=input_dirs,
                       asset_config_filename=args.asset_config_filename,
                       changed_files=changed_files)

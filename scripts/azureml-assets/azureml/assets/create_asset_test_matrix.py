# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import json
import sys
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
                       package_versions: Path,
                       changed_files: List[Path],
                       reports_dir: Path = None) -> bool:
    counters = Counter()
    asset_config_files = []
    for asset_config in util.find_assets(input_dirs, asset_config_filename, changed_files=changed_files):
        # Skip assets without testing enabled
        if not asset_config.pytest_enabled:
            logger.log_debug(f"Testing is not enabled for {asset_config}")
            continue

        # TODO: Store asset and create matrix
        logger.log_debug(f"Need to add {asset_config} to testing matrix")
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
    parser.add_argument("-p", "--package-versions-file", required=True, type=Path,
                        help="File with package versions for the base conda environment")
    parser.add_argument("-c", "--changed-files", help="Comma-separated list of changed files, used to filter assets")
    parser.add_argument("-r", "--reports-dir", type=Path, help="Directory for pytest reports")
    args = parser.parse_args()

    # Convert comma-separated values to lists
    input_dirs = [Path(d) for d in args.input_dirs.split(",")]
    changed_files = [Path(f) for f in args.changed_files.split(",")] if args.changed_files else []

    # Test assets
    create_test_matrix(input_dirs=input_dirs,
                       asset_config_filename=args.asset_config_filename,
                       package_versions=args.package_versions_file,
                       changed_files=changed_files,
                       reports_dir=args.reports_dir)

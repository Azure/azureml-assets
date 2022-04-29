import argparse
import os
import sys
from collections import defaultdict
from typing import List

from ci_logger import logger
from config import AssetConfig, AssetType, EnvironmentConfig, Spec


def validate_assets(input_dirs: List[str],
                    asset_config_filename: str):
    # Find assets under input dirs
    asset_count = 0
    error_count = 0
    asset_dirs = defaultdict(list)
    for input_dir in input_dirs:
        for root, _, files in os.walk(input_dir):
            for asset_config_file in [f for f in files if f == asset_config_filename]:
                asset_count += 1

                # Load config
                asset_config_path = os.path.join(root, asset_config_file)
                try:
                    asset_config = AssetConfig(asset_config_path)
                except Exception as e:
                    logger.log_error(f"Validation of {asset_config_path} failed: {e}")
                    error_count += 1
                    continue
                asset_dirs[f"{asset_config.type.value} {asset_config.name}"].append(asset_config_path)

                # Validate specific asset types
                if asset_config.type is AssetType.ENVIRONMENT:
                    try:
                        _ = EnvironmentConfig(asset_config.extra_config_with_path)
                    except Exception as e:
                        logger.log_error(f"Validation of {asset_config.extra_config_with_path} failed: {e}")
                        error_count += 1

                # Validate spec
                try:
                    _ = Spec(asset_config.spec_with_path)
                except Exception as e:
                    logger.log_error(f"Validation of {asset_config.spec_with_path} failed: {e}")
                    error_count += 1

    # Ensure unique assets
    for type_and_name, dirs in asset_dirs.items():
        if len(dirs) > 1:
            logger.log_error(f"{type_and_name} found in multiple asset YAMLs: {dirs}")
            error_count += 1

    print(f"Found {error_count} error(s) in {asset_count} asset(s)")
    if error_count > 0:
        sys.exit(1)


if __name__ == '__main__':
    # Handle command-line args
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-dirs", required=True, help="Comma-separated list of directories containing assets")
    parser.add_argument("-a", "--asset-config-filename", default="asset.yaml", help="Asset config file name to search for")
    args = parser.parse_args()

    # Convert comma-separated values to lists
    input_dirs = args.input_dirs.split(",")

    # Validate assets
    validate_assets(input_dirs=input_dirs,
                  asset_config_filename=args.asset_config_filename)

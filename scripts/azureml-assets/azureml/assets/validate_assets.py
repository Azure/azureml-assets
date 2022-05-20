import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import List

import azureml.assets as assets
import azureml.assets.util as util
from azureml.assets.config import ValidationException
from azureml.assets.util import logger


def validate_assets(input_dirs: List[Path],
                    asset_config_filename: str) -> bool:
    # Find assets under input dirs
    asset_count = 0
    error_count = 0
    asset_dirs = defaultdict(list)
    for asset_config_path in util.find_asset_config_files(input_dirs, asset_config_filename):
        asset_count += 1

        # Load config
        try:
            asset_config = assets.AssetConfig(asset_config_path)
        except Exception as e:
            logger.log_error(f"Validation of {asset_config_path} failed: {e}")
            error_count += 1
            continue
        asset_dirs[f"{asset_config.type.value} {asset_config.name}"].append(asset_config_path)

        # Validate specific asset types
        if asset_config.type is assets.AssetType.ENVIRONMENT:
            try:
                _ = assets.EnvironmentConfig(asset_config.extra_config_with_path)
            except Exception as e:
                logger.log_error(f"Validation of {asset_config.extra_config_with_path} failed: {e}")
                error_count += 1

        # Validate spec
        try:
            spec = assets.Spec(asset_config.spec_with_path)

            # Ensure name and version aren't inconsistent
            if not assets.Config._contains_template(spec.name) and asset_config.name != spec.name:
                raise ValidationException(f"Asset and spec name mismatch: {asset_config.name} != {spec.name}")
            if not assets.Config._contains_template(spec.version) and asset_config.version != spec.version:
                raise ValidationException(f"Asset and spec version mismatch: {asset_config.version} != {spec.version}")
        except Exception as e:
            logger.log_error(f"Validation of {asset_config.spec_with_path} failed: {e}")
            error_count += 1

    # Ensure unique assets
    for type_and_name, dirs in asset_dirs.items():
        if len(dirs) > 1:
            logger.log_error(f"{type_and_name} found in multiple asset YAMLs: {dirs}")
            error_count += 1

    print(f"Found {error_count} error(s) in {asset_count} asset(s)")
    return error_count == 0


if __name__ == '__main__':
    # Handle command-line args
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-dirs", required=True, help="Comma-separated list of directories containing assets")
    parser.add_argument("-a", "--asset-config-filename", default=assets.DEFAULT_ASSET_FILENAME, help="Asset config file name to search for")
    args = parser.parse_args()

    # Convert comma-separated values to lists
    input_dirs = [Path(d) for d in args.input_dirs.split(",")]

    # Validate assets
    success = validate_assets(input_dirs=input_dirs,
                              asset_config_filename=args.asset_config_filename)
    if not success:
        sys.exit(1)

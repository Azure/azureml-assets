import argparse
from pathlib import Path

import azureml.assets as assets
import azureml.assets.util as util
from azureml.assets.util import logger

COPIED_COUNT = "copied_count"


def copy_unreleased_asset(asset_config: assets.AssetConfig,
                          release_directory_root: Path,
                          output_directory_root: Path) -> str:
    if assets.release_tag_exists(asset_config, release_directory_root):
        # Skip a released version
        return None

    # Copy asset to output directory
    util.copy_asset_to_output_dir(asset_config, output_directory_root)
    return assets.Spec(asset_config.spec_with_path).version


def copy_unreleased_assets(release_directory_root: Path,
                           output_directory_root: Path,
                           asset_config_filename: str):
    # Find assets under release dir
    asset_count = 0
    copied_count = 0
    for asset_config in util.find_assets(release_directory_root, asset_config_filename):
        asset_count += 1

        # Copy asset if tag doesn't exist
        version = copy_unreleased_asset(asset_config=asset_config,
                                        release_directory_root=release_directory_root,
                                        output_directory_root=output_directory_root)
        if version:
            print(f"Copied {asset_config.type.value} {asset_config.name} version {version}")
            copied_count += 1
    print(f"{copied_count} of {asset_count} asset(s) copied")

    # Set variables
    logger.set_output(COPIED_COUNT, copied_count)


if __name__ == '__main__':
    # Handle command-line args
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--release-directory", required=True, type=Path, help="Directory to which the release branch has been cloned")
    parser.add_argument("-o", "--output-directory", required=True, type=Path, help="Directory to which unreleased assets will be written")
    parser.add_argument("-a", "--asset-config-filename", default=assets.DEFAULT_ASSET_FILENAME, help="Asset config file name to search for")
    args = parser.parse_args()

    # Release assets
    copy_unreleased_assets(release_directory_root=args.release_directory,
                           output_directory_root=args.output_directory,
                           asset_config_filename=args.asset_config_filename)

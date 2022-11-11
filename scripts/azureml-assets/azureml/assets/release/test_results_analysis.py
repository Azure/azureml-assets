# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Python scripts for comparing test results with next stage config."""
from pathlib import Path
import yaml
from azureml.assets.util import logger
import argparse
import azureml.assets as assets
import azureml.assets.util as util
SUPPORTED_ASSET_TYPES = [assets.AssetType.COMPONENT.value]


def test_results_analysis(config_file: Path, results_file: Path, asset_dir: Path):
    """Compare test results with create list."""
    valid_assets = []
    with open(results_file) as fp:
        valid_assets = yaml.load(fp, Loader=yaml.FullLoader)

    covered_assets = []
    uncovered_assets = []
    with open(config_file) as fp:
        config = yaml.load(fp, Loader=yaml.FullLoader)
        create_list = config.get('create', {})
        for assets_type in create_list:
            if assets_type not in SUPPORTED_ASSET_TYPES:
                continue
            assets_list = create_list[assets_type]
            if '*' in assets_list:
                assets_set = util.find_assets(
                    input_dirs=asset_dir,
                    types=assets.AssetType(assets_type))
                assets_list = [asset.name for asset in assets_set]
                logger.print(f"find all {assets_type}s: {assets_list}")
            for asset in assets_list:
                if asset in valid_assets:
                    covered_assets.append(asset)
                else:
                    uncovered_assets.append(asset)

    logger.print(f"covered assets: {covered_assets}")
    if len(uncovered_assets) > 0:
        logger.log_warning(
            "Not all assets in next stage create list are covered by completed test jobs."
            f"Uncovered assets: {uncovered_assets}.")


if __name__ == '__main__':
    # Handle command-line args
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config-file", type=Path, required=True, help="path of next stage config")
    parser.add_argument("-r", "--results-file", type=Path, required=False, help="path of test results")
    parser.add_argument("-a", "--assets-directory", required=True, type=Path, help="the assets directory")
    args = parser.parse_args()
    config_file = args.config_file
    test_results = args.results_file
    asset_folder = args.assets_directory

    test_results_analysis(config_file, test_results, asset_folder)

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Python scripts for comparing test results with next stage config."""
from pathlib import Path
import yaml
from azureml.assets.util import logger
import argparse


def test_results_analysis(config_file:Path, results_file:Path):
    """Compare test results with create list"""
    valid_assets=[]
    with open(results_file) as fp:
        valid_assets = yaml.load(fp, Loader=yaml.FullLoader)
        
    covered_assets = []
    uncovered_assets = []
    with open(config_file) as fp:
        config = yaml.load(fp, Loader=yaml.FullLoader)
        create_list = config.get('create', {})
        for assets_list in create_list.values():
            for asset in  assets_list:
                if asset in valid_assets:
                    covered_assets.append(asset)
                else: 
                    uncovered_assets.append(asset)
                
    if len(uncovered_assets)>0:
        logger.log_error(f"Not all assets in next stage create list are covered by completed test jobs. Uncovered assets: {uncovered_assets}.")


if __name__ == '__main__':
    # Handle command-line args
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config-file", type=Path, required=True, help="path of next stage config")
    parser.add_argument("-r", "--results-file", type=Path, required=False, help="path of test results")
    args = parser.parse_args()
    config_file= args.config_file
    test_results = args.results_file
    
    test_results_analysis(config_file, test_results)
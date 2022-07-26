# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from pathlib import Path
import yaml
import shutil
import argparse
import azureml.assets as assets
import azureml.assets.util as util
from azureml.assets.util import logger


def copy_replace_dir(source: Path, dest: Path):
    """Copy and replace the source dir to dest dir."""
    if dest.exists():
        shutil.rmtree(dest)
    # Copy source to destination directory
    shutil.copytree(source, dest)


def process_test_files(src_yaml: Path, assets_name_list: list):
    """Process the test files and replace the local reference to the asset with the asset name for later use."""
    covered_assets = []
    with open(src_yaml) as fp:
        data = yaml.load(fp, Loader=yaml.FullLoader)
        for test_group in data.values():
            for test_job in test_group['jobs'].values():
                test_job_path = src_yaml.parent / test_job['job']
                with open(test_job_path) as tj:
                    tj_yaml = yaml.load(tj, Loader=yaml.FullLoader)
                    for job_name, job in tj_yaml["jobs"].items():
                        if job["component"].split(":")[0] == 'file':
                        # only process the local file asset
                            original_asset = job["component"].split(":")[1]
                            asset_folder = test_job_path.parent / Path(original_asset).parent
                            local_assets = util.find_assets(
                                input_dirs=asset_folder,
                                asset_config_filename=assets.DEFAULT_ASSET_FILENAME)    
                            if len(local_assets) > 0 and local_assets[0].name in assets_name_list:
                                job["component"] = local_assets[0].name
                                covered_assets.append(local_assets[0].name)
                                logger.print(f"for job {job_name}, find Asset name: {job['component']}")
                with open(test_job_path, "w") as file:
                    yaml.dump(
                        tj_yaml,
                        file,
                        default_flow_style=False,
                        sort_keys=False)
    return covered_assets


if __name__ == '__main__':
    # Handle command-line args
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-dir", required=True, type=Path,
                        help="dir path of tests.yml")
    parser.add_argument("-a", "--test-area", required=True, type=str,
                        help="the test area name")
    parser.add_argument(
        "-r",
        "--release-directory",
        required=True,
        type=Path,
        help="Directory to which the release branch has been cloned")
    args = parser.parse_args()
    yaml_name = "tests.yml"
    # supported asset types could be extended in the future
    supported_asset_types = [assets.AssetType.COMPONENT]
    tests_folder = args.release_directory / "tests" / args.test_area
    Path.mkdir(tests_folder, parents=True, exist_ok=True)
    src_dir = args.input_dir
    assets_list = util.find_assets(src_dir, assets.DEFAULT_ASSET_FILENAME)
    assets_names = [asset.name for asset in assets_list if asset.type in supported_asset_types]
    src_yaml = src_dir / yaml_name
    covered_assets = process_test_files(src_yaml, assets_names)
    uncovered_assets = [asset for asset in assets_names if asset not in covered_assets]
    if len(uncovered_assets) > 0:
        logger.log_warning(f"The following assets are not covered by the test: {uncovered_assets}")
    shutil.copy(src_yaml, tests_folder / yaml_name)

    with open(src_yaml) as fp:
        data = yaml.load(fp, Loader=yaml.FullLoader)
        for test_group in data:
            for include_file in data[test_group].get('includes', []):
                target_path = tests_folder / include_file
                src_path = src_dir / include_file
                if (src_path).is_dir():
                    logger.print(f"copying folder: {include_file}")
                    copy_replace_dir(src_path, target_path)
                else:
                    logger.print(f"copying file: {include_file}")
                    Path.mkdir(
                        target_path.parent,
                        parents=True,
                        exist_ok=True)
                    shutil.copy(src_path, target_path)

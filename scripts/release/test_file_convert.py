# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Python scripts for test files converting in GitHub Actions."""
from pathlib import Path
import yaml
import shutil
import argparse
import azureml.assets as assets
import azureml.assets.util as util
from azureml.assets.util import logger
from typing import List, Tuple, Union

EXCLUDE_DIR_PREFIX = "!"
TEST_YAML_NAME = "tests.yml"


def copy_replace_dir(source: Path, dest: Path):
    """Copy and replace the source dir to dest dir."""
    if dest.exists():
        shutil.rmtree(dest)
    # Copy source to destination directory
    shutil.copytree(source, dest)


def process_test_files(src_yaml: Path, assets_name_list: list):
    """Process the test files and replace the local reference to the asset with the asset name for later use."""
    all_covered_assets = []
    with open(src_yaml) as fp:
        data = yaml.load(fp, Loader=yaml.FullLoader)
    for test_group in data.values():
        for test_job in test_group['jobs'].values():
            covered_assets = []
            if "pytest_job" in test_job:
                for asset_path in test_job["assets"]:
                    asset_config = util.find_assets(input_dirs=src_yaml.parent / asset_path)[0]
                    covered_assets.append(asset_config.name)
            else:
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

            test_job["assets"] = covered_assets
            # save test job yaml
            with open(test_job_path, "w") as file:
                yaml.dump(tj_yaml, file, default_flow_style=False, sort_keys=False)
            # save tests.yaml
            with open(src_yaml, "w") as file:
                yaml.dump(data, file, default_flow_style=False, sort_keys=False)
            all_covered_assets.extend(covered_assets)
    return all_covered_assets


def _convert_excludes(input_dirs: Union[List[Path], Path],
                      exclude_dirs: List[Path] = None,
                      working_root: Path = Path(".")) -> Tuple[List[Path], List[Path]]:
    """Extract directories to exclude from input_dirs and add them to exclude_dirs."""
    if type(input_dirs) is not list:
        input_dirs = [input_dirs]
    if exclude_dirs is not None:
        if type(exclude_dirs) is not list:
            exclude_dirs = [exclude_dirs]
    else:
        exclude_dirs = []

    # Exclude any dirs that start with EXCLUDE_DIR_PREFIX
    new_input_dirs = []
    new_exclude_dirs = []

    for input_dir in input_dirs:
        input_dir_str = str(input_dir)
        if input_dir_str.startswith(EXCLUDE_DIR_PREFIX):
            new_exclude_dirs.append(working_root / Path(input_dir_str[len(EXCLUDE_DIR_PREFIX):]))
        elif input_dir_str == '.':
            for current_folder in working_root.iterdir():
                if current_folder.is_dir() and current_folder not in exclude_dirs:
                    new_input_dirs.append(current_folder)
        else:
            new_input_dirs.append(working_root / input_dir)

    if new_exclude_dirs:
        if exclude_dirs:
            exclude_dirs.extend(new_exclude_dirs)
        else:
            exclude_dirs = new_exclude_dirs

    return new_input_dirs, exclude_dirs


if __name__ == '__main__':
    # Handle command-line args
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-dirs", required=True, help="dir path of test areas")
    parser.add_argument("-w", "--working-root", type=Path, required=False, help="dir path of working root")
    parser.add_argument("-r", "--release-directory", required=True, type=Path,
                        help="Directory to which the release branch has been cloned")
    args = parser.parse_args()
    # Convert comma-separated values to lists

    working_root = args.working_root if args.working_root else Path(".")
    input_dirs = [Path(d) for d in args.input_dirs.split(",")]
    src_dirs, exclude_dirs = _convert_excludes(input_dirs, working_root=working_root)

    for src_dir in src_dirs:
        if src_dir in exclude_dirs:
            continue
        test_area = src_dir.name
        logger.print(f"Processing test area: {test_area}")

        if not (src_dir / TEST_YAML_NAME).exists():
            logger.log_warning(f"Cannot find {TEST_YAML_NAME} in the test area {test_area}.")
            continue

        # supported asset types could be extended in the future
        supported_asset_types = [assets.AssetType.COMPONENT]
        tests_folder = args.release_directory / "tests" / test_area
        Path.mkdir(tests_folder, parents=True, exist_ok=True)

        assets_list = util.find_assets(src_dir, assets.DEFAULT_ASSET_FILENAME)
        assets_names = [asset.name for asset in assets_list if asset.type in supported_asset_types]
        src_yaml = src_dir / TEST_YAML_NAME
        covered_assets = process_test_files(src_yaml, assets_names)
        uncovered_assets = [asset for asset in assets_names if asset not in covered_assets]
        if len(uncovered_assets) > 0:
            logger.log_warning(f"The following assets are not covered by the test: {uncovered_assets}")
        shutil.copy(src_yaml, tests_folder / TEST_YAML_NAME)

        with open(src_yaml) as fp:
            data = yaml.load(fp, Loader=yaml.FullLoader)
            for test_group in data:
                for include_file in data[test_group].get('includes', []):
                    target_path = tests_folder / include_file
                    src_path = src_dir / include_file
                    if src_path.is_dir():
                        logger.print(f"copying folder: {include_file}")
                        copy_replace_dir(src_path, target_path)
                    else:
                        logger.print(f"copying file: {include_file}")
                        Path.mkdir(
                            target_path.parent,
                            parents=True,
                            exist_ok=True)
                        shutil.copy(src_path, target_path)

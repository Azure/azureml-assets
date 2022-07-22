# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""python scripts for test files converting in GitHub Actions"""
from pathlib import Path
import yaml
import shutil
import argparse
import azureml.assets as assets
import azureml.assets.util as util


def copy_replace_dir(source: Path, dest: Path):
    """copy and replace the source dir to dest dir"""
    if dest.exists():
        shutil.rmtree(dest)
    # Copy source to destination directory
    shutil.copytree(source, dest)


def process_test_files(src_yaml: Path):
    """process the test files and replace the local reference to the asset with the asset name for later use"""
    with open(src_yaml) as fp:
        data = yaml.load(fp, Loader=yaml.FullLoader)
        for test_group in data.values():
            for test_job in test_group['jobs'].values():
                test_job_path = src_yaml.parent / test_job['job']
                with open(test_job_path) as tj:
                    tj_yaml = yaml.load(tj, Loader=yaml.FullLoader)
                    for job in tj_yaml["jobs"]:
                        original_asset = tj_yaml["jobs"][job]["component"].split(":")[1]
                        if Path(original_asset).stem == 'spec':
                            print(test_job_path.parent / original_asset)
                            asset_folder = test_job_path.parent / Path(original_asset).parent
                            asset_name = util.find_assets(
                                input_dirs=asset_folder,
                                asset_config_filename=assets.DEFAULT_ASSET_FILENAME)[0].name
                            tj_yaml["jobs"][job]["component"] = asset_name
                            print(
                                f"Find Asset name: {tj_yaml['jobs'][job]['component']}")
                with open(test_job_path, "w") as file:
                    yaml.dump(
                        tj_yaml,
                        file,
                        default_flow_style=False,
                        sort_keys=False)


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
    tests_folder = args.release_directory / "tests" / args.test_area
    Path.mkdir(tests_folder, parents=True, exist_ok=True)
    src_dir = args.input_dir
    src_yaml = src_dir / yaml_name
    process_test_files(src_yaml)
    shutil.copy(src_yaml, tests_folder / yaml_name)

    with open(src_yaml) as fp:
        data = yaml.load(fp, Loader=yaml.FullLoader)
        for test_group in data:
            for include_file in data[test_group].get('includes', []):
                target_path = tests_folder / include_file
                if (src_dir / include_file).is_dir():
                    print(f"copying folder: {include_file}")
                    copy_replace_dir(src_dir / include_file, target_path)
                else:
                    print(f"copying file: {include_file}")
                    Path.mkdir(
                        target_path .parent,
                        parents=True,
                        exist_ok=True)
                    shutil.copy(src_dir / include_file, target_path)

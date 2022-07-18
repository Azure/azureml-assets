# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
from pathlib import Path
import yaml
import shutil
import argparse


def copy_replace_dir(source: Path, dest: Path):
    # Delete destination directory
    if dest.exists():
        shutil.rmtree(dest)
    # Copy source to destination directory
    shutil.copytree(source, dest)


if __name__ == '__main__':
    # Handle command-line args
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-dir", required=True, type=Path, help="dir path of tests.yml")
    parser.add_argument("-a", "--test-area", required=True, type=str, help="the test area name")
    parser.add_argument("-r", "--release-directory", required=True, type=Path, help="Directory to which the release branch has been cloned")
    args = parser.parse_args()
    yaml_name = "tests.yml"
    tests_folder = args.release_directory / "tests" / args.test_area
    Path.mkdir(tests_folder, parents=True, exist_ok=True)
    src_dir = args.input_dir
    src_yaml = src_dir / yaml_name
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
                    Path.mkdir(target_path .parent, parents=True, exist_ok=True)
                    shutil.copy(src_dir / include_file, target_path)

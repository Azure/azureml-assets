from asyncio import subprocess
from pathlib import Path
import sys
import yaml
import shutil
import argparse

def copy_replace_dir(source: Path, dest: Path):
    # Delete destination directory
    if dest.exists():
        shutil.rmtree(dest)
    # Copy source to destination directory
    shutil.copytree(source, dest)

# Handle command-line args
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input-dir", required=True, type=Path, help="dir path of tests.yml")
parser.add_argument("-a", "--test-area", required=True, type=str, help="the test area name")
parser.add_argument("-r", "--release-directory", required=True, type=Path, help="Directory to which the release branch has been cloned")
args = parser.parse_args()
yaml_name = "tests.yml"
tests_folder_dir = args.release_directory / "tests" / args.test_area
Path.mkdir(tests_folder_dir, parents=True, exist_ok=True)
src_dir = args.input_dir
shutil.copy(src_dir / yaml_name , tests_folder_dir / yaml_name)

with open(src_dir/ yaml_name) as fp:
    data = yaml.load(fp, Loader=yaml.FullLoader)
    for test_group in data:
        for include_file in data[test_group]['includes']:
            if (src_dir / include_file).is_dir():
                print("copying folder: "+include_file)
                copy_replace_dir((src_dir / include_file), (tests_folder_dir / include_file))
            else:
                print("copying file: "+include_file)
                Path.mkdir(tests_folder_dir / Path(include_file).parent, parents=True, exist_ok=True)
                shutil.copy((src_dir / include_file), (tests_folder_dir / include_file))


            
        
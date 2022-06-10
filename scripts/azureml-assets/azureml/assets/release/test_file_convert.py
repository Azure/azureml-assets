from asyncio import subprocess
from pathlib import Path
import os
import sys
import ruamel.yaml
import shutil
import argparse
def copy_replace_file(source: Path, dest: Path):
    # Delete destination directory
    if dest.exists():
        dest.unlink()

    # Copy source to destination directory
    shutil.copy(source, dest)
def find_folder(path:str):
    return '/'.join(path.split('/')[0:-1])
# Handle command-line args
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input-dir", required=True, type=Path, help="dir path of tests.yml")
parser.add_argument("-a", "--test-area", required=True, type=str, help="the test area name")
parser.add_argument("-r", "--release-directory", required=True, type=Path, help="Directory to which the release branch has been cloned")
args = parser.parse_args()
area_name = args.test_area  
tests_folder_dir = args.release_directory.__str__()+"/tests/"+area_name
if os.path.exists(tests_folder_dir) == False:
    os.makedirs(tests_folder_dir)
src_dir = args.input_dir

copy_replace_file(Path(src_dir.__str__()+"/tests.yml"), Path(tests_folder_dir.__str__()+'/tests.yml'))

yaml = ruamel.yaml.YAML()
with open(src_dir.__str__()+"/tests.yml") as fp:
    data = yaml.load(fp)
    for test_group in data:
        for include_file in data[test_group]['includes']:
            print(find_folder(include_file))
            if os.path.exists(Path(tests_folder_dir.__str__()+'/'+find_folder(include_file))) == False:
                os.makedirs(Path(tests_folder_dir.__str__()+'/'+find_folder(include_file)))
            copy_replace_file(Path(src_dir.__str__()+"/"+include_file), Path(tests_folder_dir.__str__()+'/'+include_file))
        
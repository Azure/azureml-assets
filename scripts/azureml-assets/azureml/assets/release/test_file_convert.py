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
tests_folder_dir = args.release_directory / "tests" / args.test_area
Path.mkdir(tests_folder_dir, parents=True, exist_ok=True)
src_dir = args.input_dir
copy_replace_file(Path(src_dir.__str__()+"/tests.yml"), Path(tests_folder_dir.__str__()+'/tests.yml'))

yaml = ruamel.yaml.YAML()
with open(src_dir.__str__()+"/tests.yml") as fp:
    data = yaml.load(fp)
    for test_group in data:
        for include_file in data[test_group]['includes']:
            if os.path.isfile(Path(src_dir.__str__()+"/"+include_file)):
                print("copying file: "+include_file)
                if os.path.exists(Path(tests_folder_dir.__str__()+'/'+find_folder(include_file))) == False:
                    os.makedirs(Path(tests_folder_dir.__str__()+'/'+find_folder(include_file)))
                copy_replace_file(Path(src_dir.__str__()+"/"+include_file), Path(tests_folder_dir.__str__()+'/'+include_file))
            else:
                print("copying folder: "+include_file)
                copy_replace_dir(Path(src_dir.__str__()+"/"+include_file), Path(tests_folder_dir.__str__()+'/'+include_file))


            
        
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

# Handle command-line args
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input-dir", required=True, type=Path, help="dir path of tests.yml")
parser.add_argument("-r", "--release-directory", required=True, type=Path, help="Directory to which the release branch has been cloned")
args = parser.parse_args()

scripts_folder_dir = args.release_directory.__str__()+"/scripts"
if os.path.exists(scripts_folder_dir) == False:
    os.makedirs(scripts_folder_dir)
src_dir = args.input_dir
src_name = ''
if '/' in src_dir.__str__() :
    src_name = src_dir.__str__().split('/')[-1]
else :
    src_name = src_dir.__str__().split('\\')[-1]
copy_replace_file(Path(src_dir), Path(scripts_folder_dir+'/'+src_name))
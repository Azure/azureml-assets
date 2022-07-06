from asyncio import subprocess
from pathlib import Path
import sys
import shutil
import argparse

# Handle command-line args
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input-dir", required=True, type=Path, help="dir path of tests.yml")
parser.add_argument("-r", "--release-directory", required=True, type=Path, help="Directory to which the release branch has been cloned")
args = parser.parse_args()

scripts_folder_dir = args.release_directory / "scripts"
Path.mkdir(scripts_folder_dir, parents=True, exist_ok=True)
src_dir = args.input_dir
src_name = ''
if '/' in src_dir.__str__() :
    src_name = src_dir.__str__().split('/')[-1]
else :
    src_name = src_dir.__str__().split('\\')[-1]
shutil.copy(src_dir, scripts_folder_dir / src_name)
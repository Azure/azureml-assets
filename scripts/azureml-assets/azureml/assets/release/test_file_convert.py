from pathlib import Path
import os
import sys
import ruamel.yaml
import shutil

tests_folder_dir = sys.argv[1]

src_dir = Path("../../../../../vision/src")

yaml = ruamel.yaml.YAML()
with open(src_dir.__str__()+"/tests.yml") as fp:
    data = yaml.load(fp)
    for job in data:
        test_file_dir = src_dir.__str__()+job["path"]
        shutil.copyfile(test_file_dir, tests_folder_dir.__str__()+"/"+job)
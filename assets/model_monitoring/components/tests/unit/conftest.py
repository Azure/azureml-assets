#Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Unit test level fixtures."""

import os
import pytest
import zipfile
import time
import random
import string
from spark_mltable import SPARK_ZIP_PATH


@pytest.fixture(scope="session")
def code_zip_test_setup():
    """Zip files in module_path to src.zip."""
    momo_work_dir = os.path.abspath(f"{os.path.dirname(__file__)}/../..")
    module_path = os.path.join(momo_work_dir, "src")
    # zip files in module_path to src.zip
    s = string.ascii_lowercase + string.digits
    rand_str = '_' + ''.join(random.sample(s, 5))
    time_str = time.strftime("%Y%m%d-%H%M%S") + rand_str
    zip_path = os.path.join(module_path, f"src_{time_str}.zip")

    zf = zipfile.ZipFile(zip_path, "w")
    for dirname, subdirs, files in os.walk(module_path):
        for filename in files:
            abs_filepath = os.path.join(dirname, filename)
            rel_filepath = os.path.relpath(abs_filepath, start=module_path)
            if not rel_filepath.endswith(".py"):
                print("skipping file:", rel_filepath)
                continue
            print("zipping file:", rel_filepath)
            zf.write(abs_filepath, arcname=rel_filepath)
    zf.close()
    # add files to zip folder
    os.environ[SPARK_ZIP_PATH] = zip_path
    print("zip path set in code_zip_test_setup: ", zip_path)

    yield
    # remove zip file
    os.remove(zip_path)
    # remove zip path from environment
    os.environ.pop(SPARK_ZIP_PATH, None)
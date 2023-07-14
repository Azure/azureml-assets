# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Module init file."""

import os
import sys
from pathlib import Path

# Adding model monitoring component source to path.
src_path = os.path.join(Path(os.path.dirname(__file__)).parents[0], "src", "")
sys.path.append(src_path)

dir_list = os.listdir(src_path)
for file in dir_list:
    sub_path = os.path.join(src_path, file, "")
    sys.path.append(sub_path)

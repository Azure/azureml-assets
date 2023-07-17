# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Module init file."""

import os
import sys
from pathlib import Path

# Adding model monitoring component source to path.
src_path = os.path.join(Path(os.path.dirname(__file__)).parents[0], "src", "")
sys.path.append(src_path)

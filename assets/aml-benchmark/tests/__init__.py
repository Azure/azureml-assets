# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Init file."""

import sys

from .test_utils import get_src_dir


path = get_src_dir()
if path not in sys.path:
    sys.path.append(path)

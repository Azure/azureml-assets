# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Batch benchmark score init file."""

import sys
import os


path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if path not in sys.path:
    sys.path.append(path)

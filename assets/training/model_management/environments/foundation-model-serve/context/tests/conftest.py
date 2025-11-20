# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import sys


def pytest_configure(config):
    current_directory = os.path.dirname(os.path.abspath(__file__))
    parent_directory = os.path.dirname(current_directory)
    src_directory = os.path.join(parent_directory, "context", "foundation", "model", "serve")

    # Need to add src directory to the path to enable discovery of src files by the test directory
    sys.path.append(parent_directory)
    sys.path.append(src_directory)

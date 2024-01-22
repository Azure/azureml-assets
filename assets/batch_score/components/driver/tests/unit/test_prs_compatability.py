# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains unit tests for PRS compatibility."""

import importlib
import mock
import os
import sys

from tests.fixtures.geneva_event_listener import mock_import

def test_prs_code_importing():
    # Should point to the same root as the "code" in the batch-score component yamls
    component_code_root = "../../src"
    file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), component_code_root))
    sys.path.append(file_path)

    # Should be the exact same as the "entry_script" in the batch-score component yamls
    entry_script = "batch_score.main"

    with mock.patch('importlib.import_module', side_effect=mock_import):
        module = importlib.import_module(entry_script, file_path)

    sys.path.pop()
    assert module
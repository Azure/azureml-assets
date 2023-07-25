# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Global test level fixtures."""

import pytest


@pytest.fixture(scope="function")
def temp_dir(tmpdir):
    """Return the path to the temporary directory as a string."""
    return str(tmpdir)

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pytest


@pytest.fixture(scope="function")
def temp_dir(tmpdir):
    # Return the path to the temporary directory as a string
    return str(tmpdir)

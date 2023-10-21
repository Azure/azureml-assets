# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Global test level fixtures."""

import os
import pytest
from utils.ComponentHelpers import generate_assets


@pytest.fixture(scope="session")
def large_language_models_root_directory() -> str:
    '''large_language_models_root_directory'''

    return os.path.abspath(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
    )


@pytest.fixture(scope="session")
def asset_lists(large_language_models_root_directory):
    '''asset_lists'''

    (comp_assets_all, pipe_assets_all) = generate_assets(large_language_models_root_directory)
    return (comp_assets_all, pipe_assets_all)

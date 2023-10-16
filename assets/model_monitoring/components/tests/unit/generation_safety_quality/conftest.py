# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


"""This file contains unit tests for the request logic."""

import os
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def test_data():
    """Load test data."""
    data = pd.read_json(
        os.path.join(
            os.getcwd(),
            "tests",
            "unit",
            "generation_safety_quality",
            "test_data_gsq.jsonl"),
        dtype=False, lines=True)
    yield data

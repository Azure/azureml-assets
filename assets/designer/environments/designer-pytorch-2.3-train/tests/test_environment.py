# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Test the Designer PyTorch train environment in a connected Azure ML workspace."""

from pathlib import Path
from runpy import run_path

RUN_SMOKE_TEST = run_path(
    str(Path(__file__).resolve().parents[2] / "test_helpers" / "environment_smoke.py")
)["run_environment_smoke_test"]


def test_environment():
    """Build the draft image and run its smoke-test job."""
    RUN_SMOKE_TEST(Path(__file__).resolve().parents[1])

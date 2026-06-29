# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests that fully exercise the evaluator source."""

from evaluator._calculator import add


def test_add():
    """Exercise the only source function so coverage is >=85%."""
    assert add(2, 3) == 5

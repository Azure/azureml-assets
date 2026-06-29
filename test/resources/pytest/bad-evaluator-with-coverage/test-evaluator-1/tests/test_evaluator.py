# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests that exercise only a small part of the evaluator source."""

from evaluator._calculator import add


def test_add():
    """Exercise only one of several source functions so coverage is <85%."""
    assert add(2, 3) == 5

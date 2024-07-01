# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for Checksum functionality used in Prompt Crafter Component."""

import sys
import pytest

from ..test_utils import get_src_dir

sys.path.append(get_src_dir())
try:
    from aml_benchmark.prompt_crafter.package.checksum import SHA256Checksum
except ImportError:
    raise ImportError("Please install the package 'prompt_crafter' to run this test.")


@pytest.fixture
def checksum1():
    """Fixture for checksum."""
    return SHA256Checksum()


@pytest.fixture
def checksum2():
    """Fixture for checksum."""
    return SHA256Checksum()


def test_different_for_different_lines(checksum1, checksum2):
    """Test that checksums are different for different lines."""
    checksum1.update({'a': 1})
    checksum2.update({'a': 2})

    assert checksum1.digest() != checksum2.digest()


def test_independent_of_key_order(checksum1, checksum2):
    """Test that checksums are independent of key order."""
    checksum1.update({'a': 1, 'b': 'x'})
    checksum2.update({'b': 'x', 'a': 1})

    assert checksum1.digest() == checksum2.digest()


def test_works_with_structures(checksum1):
    """Test that checksums work with nested structures."""
    jsonl_line = {
        'a': [1, 2, 3],
        'b': {
            'x': 'y',
            'z': [7, 8]
        }
    }
    checksum1.update(jsonl_line)
    _ = checksum1.digest()


def test_works_with_empty(checksum1):
    """Test that checksums work with empty structures."""
    checksum1.update({})
    _ = checksum1.digest()


def test_works_with_none(checksum1):
    """Test that checksums work with None."""
    checksum1.update(None)
    _ = checksum1.digest()

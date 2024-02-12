# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for common evaluation utilities."""

import pytest

from aml_benchmark.utils.evaluation_utils import extract_python_function, extract_text_from_markdown_tag


@pytest.mark.parametrize(
        'tag_open,tag_close',
        [('```python', '```'), ('```python', ''), ('', '')]
)
def test_extract_markdown_tags(tag_open, tag_close):
    python_func_text = 'def foo():\n    return 0\n'
    response = tag_open + python_func_text + tag_close
    response_parsed = extract_text_from_markdown_tag(response, tag_type='python')
    assert response_parsed == python_func_text


@pytest.mark.parametrize(
        'is_partial,extra_text',
        [(False, ''), (False, '\ndef new_foo():\n'), (True, ''), (True, '\ndef new_foo():\n')]
)
def test_extract_python_func(is_partial, extra_text):
    python_func_text = 'import os\ndef foo():\n    return 0' if not is_partial else '    return 0'
    response = python_func_text + extra_text
    response_parsed = extract_python_function(response, partial=is_partial)
    assert response_parsed == python_func_text

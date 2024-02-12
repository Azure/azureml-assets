# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# ---------------------------------------------------------

"""Common evaluation utilities for benchmark components."""

import re

def extract_text_from_markdown_tag(input_string: str, tag_type='python') -> str:
    """
    Extract text between markdown code tags.

    If the tag pattern search returns no matches, the input string is returned.
    """
    pattern = f"```{tag_type}(.*?)```"
    m = re.search(pattern, input_string, flags=re.DOTALL)
    if not m:
        pattern_partial = f"```{tag_type}(.*)"
        m = re.search(pattern_partial, input_string, flags=re.DOTALL)
    return m.group(1) if m else input_string

def extract_python_function(input_string: str, partial=False) -> str:
    """
    Extract a Python function from the input string.

    This function handles two cases: partial and full. In the partial case, we assume the model provides the body
    of the function, but not the declaration and argument list. In the full case, it's assumed the model
    provides the full function definition.

    If the code pattern search returns no matches, the input string is returned.
    """
    code_pattern = "(.*?)" if partial else "(.*?def.*?)"
    pattern = f"{code_pattern}(\nclass|\ndef|\n#|\nif|\nfor|\nwhile|\nprint|$)"
    m = re.search(pattern, input_string, flags=re.DOTALL)
    return m.group(1) if m else input_string

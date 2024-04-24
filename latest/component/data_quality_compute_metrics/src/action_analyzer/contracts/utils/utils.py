# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains utilities for action detector."""


def convert_to_camel_case(input_string: str) -> str:
    """
    Convert a snake_case string to camelCase.

    Example: "retrieval_top_k" -> "RetrievalTopK"
    """
    words = input_string.split("_")
    result = "".join(word.capitalize() for word in words)
    return result

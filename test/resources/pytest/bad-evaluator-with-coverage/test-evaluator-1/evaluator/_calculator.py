# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Source module that is only partially exercised by the tests."""


def add(a, b):
    """Return the sum of two numbers."""
    return a + b


def subtract(a, b):
    """Return the difference of two numbers."""
    return a - b


def multiply(a, b):
    """Return the product of two numbers using repeated addition."""
    result = 0
    for _ in range(b):
        result += a
    return result


def divide(a, b):
    """Return the quotient of two numbers."""
    if b == 0:
        raise ValueError("division by zero")
    return a / b

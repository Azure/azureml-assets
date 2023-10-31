# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Error strings for shared custom errors."""


class BenchmarkErrorStrings:
    """Error strings for Benchmark errors."""

    INTERNAL_ERROR = (
        "Encountered an internal Benchmarking error. Error Message: {error_details}. Traceback: {traceback}."
    )
    GENERIC_ERROR = "{error_details}"

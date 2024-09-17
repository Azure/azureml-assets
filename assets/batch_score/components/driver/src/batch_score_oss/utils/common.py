# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Common utilities."""

from argparse import ArgumentParser
from urllib.parse import urlparse


def get_base_url(url: str) -> str:
    """Get base url."""
    if not url:
        return url

    parse_result = urlparse(url)
    return f"{parse_result.scheme}://{parse_result.netloc}"


def backoff(attempt: int, base_delay: float = 1, exponent: float = 2, max_delay: float = 20):
    """Calculate backoff delay time."""
    return min(max_delay, base_delay * attempt**exponent)


def str2bool(v):
    """Convert string to boolean."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentParser.ArgumentTypeError('Boolean value expected.')


def get_mini_batch_id(mini_batch_context: any):
    """Get mini batch id from mini batch context."""
    if mini_batch_context:
        return mini_batch_context.mini_batch_id

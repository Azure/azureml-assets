# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains utilities to log a message prefixed with a timestamp."""

import logging

FORMAT = '%(asctime)s %(message)s'
logging.basicConfig(format=FORMAT)


def log(level: int, message: str):
    """Log a message of given level."""
    logging.log(level=level, msg=message)


def log_info(message: str):
    """Log an INFO message."""
    log(logging.INFO, message)


def log_warning(message: str):
    """Log a WARNING message."""
    log(logging.WARNING, message)


def log_error(message: str):
    """Log an ERROR message."""
    log(logging.ERROR, message)

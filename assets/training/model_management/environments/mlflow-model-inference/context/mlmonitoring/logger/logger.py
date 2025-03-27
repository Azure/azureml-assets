"""For logger."""
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import sys
from typing import Any, Dict
import logging
from logging.config import dictConfig

logger = logging.getLogger("mdc")

DEFAULT_SDK_LOGGING_CONFIG: Dict[str, Any] = dict(
    version=1,
    disable_existing_loggers=False,
    loggers={
        "mdc": {
            "level": "INFO",
            "propagate": False,
            "handlers": ["console"]
        },
        "mdc.root": {
            "level": "INFO",
            "propagate": False,
            "handlers": ["console"]
        },
        "mdc.collector": {
            "level": "INFO",
            "propagate": False,
            "handlers": ["console"]
        },
        "mdc.worker": {
            "level": "INFO",
            "propagate": False,
            "handlers": ["console"]
        },
        "local.capture": {
            "level": "INFO",
            "propagate": False,
            "handlers": ["console"]
        },
        "mdc.sender": {
            "level": "WARN",
            "propagate": False,
            "handlers": ["console"]
        },
        "mdc.error": {
            "level": "ERROR",
            "handlers": ["error_console"],
            "propagate": False,
            "qualname": "mdc.error",
        },
    },
    handlers={
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "generic",
            "stream": sys.stdout,
        },
        "error_console": {
            "class": "logging.StreamHandler",
            "formatter": "generic",
            "stream": sys.stderr,
        },
    },
    formatters={
        "generic": {
            "format": "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            # "datefmt": "%Y-%m-%d %H:%M:%S",
            "class": "logging.Formatter",
        },
    },
)

DEBUG_SDK_LOGGING_CONFIG: Dict[str, Any] = dict(
    version=1,
    disable_existing_loggers=False,
    loggers={
        "mdc": {
            "level": "DEBUG",
            "propagate": False,
            "handlers": ["console"]
        },
        "mdc.root": {
            "level": "DEBUG",
            "propagate": False,
            "handlers": ["console"]
        },
        "mdc.collector": {
            "level": "DEBUG",
            "propagate": False,
            "handlers": ["console"]
        },
        "mdc.worker": {
            "level": "DEBUG",
            "propagate": False,
            "handlers": ["console"]
        },
        "local.capture": {
            "level": "DEBUG",
            "propagate": False,
            "handlers": ["console"]
        },
        "mdc.sender": {
            "level": "INFO",
            "propagate": False,
            "handlers": ["console"]
        },
        "mdc.error": {
            "level": "ERROR",
            "handlers": ["error_console"],
            "propagate": False,
            "qualname": "mdc.error",
        },
    },
    handlers={
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "generic",
            "stream": sys.stdout,
        },
        "error_console": {
            "class": "logging.StreamHandler",
            "formatter": "generic",
            "stream": sys.stderr,
        },
    },
    formatters={
        "generic": {
            "format": "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            # "datefmt": "%Y-%m-%d %H:%M:%S",
            "class": "logging.Formatter",
        },
    },
)


def is_debug():
    """For is debug."""
    return os.getenv("AZUREML_MDC_DEBUG", "false").lower() == "true"


def init_logging():
    """For init logging."""
    if is_debug():
        log_config = DEBUG_SDK_LOGGING_CONFIG
    else:
        log_config = DEFAULT_SDK_LOGGING_CONFIG

    dictConfig(log_config)

# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""Foundation Model Serve classes."""
import logging
import sys

logger = logging.getLogger(__name__)
logger.propagate = False
logger.setLevel(logging.DEBUG)
format_str = "%(asctime)s [%(module)s] %(funcName)s " "%(lineno)s: %(levelname)-8s [%(process)d] %(message)s"
formatter = logging.Formatter(format_str)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

__path__ = __import__('pkgutil').extend_path(__path__, __name__)

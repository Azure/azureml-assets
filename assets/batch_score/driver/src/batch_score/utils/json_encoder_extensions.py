# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Json encoder extensions."""

import json

import numpy

from ..common.telemetry import logging_utils as lu


class BatchComponentJSONEncoderConfiguration():
    """Batch component JSON encoder configuration."""

    def __init__(self, ensure_ascii: bool) -> None:
        """Init function."""
        global _default_encoder_configuration

        if _default_encoder_configuration:
            return

        self.ensure_ascii = ensure_ascii


class BatchComponentJSONEncoder(json.JSONEncoder):
    """Batch component Json encoder."""

    def __init__(self, *,
                 skipkeys: bool = None,
                 ensure_ascii: bool = None,
                 check_circular: bool = None,
                 allow_nan: bool = None,
                 sort_keys: bool = None,
                 indent=None,
                 separators=None,
                 default=None) -> None:
        """Init function."""
        global _default_encoder_configuration
        if _default_encoder_configuration:
            super().__init__(
                ensure_ascii=_default_encoder_configuration.ensure_ascii
            )
        else:
            lu.get_logger().debug("No JSONEncoder configured, falling back to default")
            super().__init__()


class NumpyArrayEncoder(BatchComponentJSONEncoder):
    """Numpy array encoder."""

    def default(self, obj):
        """Encode the object as a numpy ND array."""
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return super().default(self, obj)


# Global module variable to keep track of state
_default_encoder_configuration: BatchComponentJSONEncoderConfiguration = None


def setup_encoder(ensure_ascii: bool = True):
    """Set up encoder."""
    global _default_encoder_configuration
    if not _default_encoder_configuration:
        _default_encoder_configuration = BatchComponentJSONEncoderConfiguration(
            ensure_ascii=ensure_ascii
        )


def get_default_encoder() -> json.JSONEncoder:
    """Get default encoder."""
    return BatchComponentJSONEncoder()

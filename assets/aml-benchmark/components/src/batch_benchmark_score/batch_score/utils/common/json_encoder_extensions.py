# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""The class for json encoder."""

import json
import numpy
from .. import logging_utils as lu


class BatchComponentJSONEncoderConfiguration():
    """Class for batch component json encoder."""

    def __init__(self, ensure_ascii: bool) -> None:
        """Init method."""
        global _default_encoder_configuration

        if _default_encoder_configuration:
            return

        self.ensure_ascii = ensure_ascii


class BatchComponentJSONEncoder(json.JSONEncoder):
    """Batch component JSON Encoder class."""

    def __init__(
            self,
            *,
            skipkeys: bool = None, ensure_ascii: bool = None,
            check_circular: bool = None, allow_nan: bool = None,
            sort_keys: bool = None, indent: int = None,
            separators: str = None, default: str = None
    ) -> None:
        """Init class for JSON encoder class."""
        global _default_encoder_configuration
        if _default_encoder_configuration:
            lu.get_logger().info("JSONEncoder configured, using configuration")
            super().__init__(
                ensure_ascii=_default_encoder_configuration.ensure_ascii
            )
        else:
            lu.get_logger().info("No JSONEncoder configured, falling back to default")
            super().__init__()


class NumpyArrayEncoder(BatchComponentJSONEncoder):
    """Numpy array encoder class."""

    def default(self, obj):
        """Override the `default` method."""
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return super().default(self, obj)


# Global module variable to keep track of state
_default_encoder_configuration: BatchComponentJSONEncoderConfiguration = None


def setup_encoder(ensure_ascii: bool = True):
    """Init encoder for global use."""
    global _default_encoder_configuration
    if not _default_encoder_configuration:
        _default_encoder_configuration = BatchComponentJSONEncoderConfiguration(
            ensure_ascii=ensure_ascii
        )


def get_default_encoder() -> json.JSONEncoder:
    """Get default encoder."""
    return BatchComponentJSONEncoder()

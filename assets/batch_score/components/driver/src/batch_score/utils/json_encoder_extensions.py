import json

import numpy

from ..common.telemetry import logging_utils as lu


class BatchComponentJSONEncoderConfiguration():
    def __init__(self, ensure_ascii: bool) -> None:
        global _default_encoder_configuration

        if _default_encoder_configuration:
            return

        self.ensure_ascii = ensure_ascii

class BatchComponentJSONEncoder(json.JSONEncoder):
    def __init__(self, *, skipkeys: bool = None, ensure_ascii: bool = None, check_circular: bool = None, allow_nan: bool = None, sort_keys: bool = None, indent = None, separators = None, default = None) -> None:
        global _default_encoder_configuration
        if _default_encoder_configuration:
            super().__init__(
                ensure_ascii = _default_encoder_configuration.ensure_ascii
            )
        else:
            lu.get_logger().debug("No JSONEncoder configured, falling back to default")
            super().__init__()

class NumpyArrayEncoder(BatchComponentJSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return super().default(self, obj)

# Global module variable to keep track of state
_default_encoder_configuration: BatchComponentJSONEncoderConfiguration = None
def setup_encoder(ensure_ascii: bool = True):
    global _default_encoder_configuration
    if not _default_encoder_configuration:
        _default_encoder_configuration = BatchComponentJSONEncoderConfiguration(
            ensure_ascii=ensure_ascii
        )

def get_default_encoder() -> json.JSONEncoder:
    return BatchComponentJSONEncoder()

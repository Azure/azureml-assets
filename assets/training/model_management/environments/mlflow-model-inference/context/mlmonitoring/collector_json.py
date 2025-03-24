"""For collector json."""
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import sys
import random
from typing import Callable, Any

from .payload import PandasFrameData
from .payload.payload import build_payload
from .queue import get_queue

from .collector_base import CollectorBase
from .context import CorrelationContext, get_context

try:
    import pandas as pd
except ImportError:
    pass


def _build_log_data_by_type(data):
    """For build log data by type."""
    if 'pandas' in sys.modules and isinstance(data, pd.DataFrame):
        return PandasFrameData(data)

    raise TypeError("data type (%s) not supported, "
                    "supported types: pandas.DataFrame"
                    % type(data).__name__)


def _raise_if_exception(e: Exception):
    """For raise if exception."""
    raise e


class JsonCollector(CollectorBase):
    """For JsonCollector."""

    def __init__(
            self,
            *,
            name: str,
            on_error: Callable[[Exception], Any] = None
    ):
        """For init."""
        super().__init__("default")
        self.name = name
        if on_error:
            self.on_error = on_error
        else:
            self.on_error = _raise_if_exception

        self._validate_mdc_config()

    def _validate_mdc_config(self):
        """For validate mdc config."""
        if not self.name or len(self.name) <= 0:
            # unexpected drop
            msg = "collection name is not provided"
            self.on_error(Exception(msg))
            return False, msg

        config = self.config
        if config is None:
            # unexpected drop
            msg = "data collector is not initialized"
            self.on_error(Exception(msg))
            return False, msg

        if not config.enabled():
            # unexpected drop
            msg = "custom logging is not enabled, drop the data"
            self.on_error(Exception(msg))
            return False, msg

        if not config.collection_enabled(self.name):
            # unexpected drop
            msg = "collection {} is not enabled, drop the data".format(self.name)
            self.on_error(Exception(msg))
            return False, msg

        return True, None

    def collect(
            self,
            data,  # supported type: Union[pd.DataFrame]
            correlation_context: CorrelationContext = None) -> CorrelationContext:
        """For collect."""
        if correlation_context is None:
            correlation_context = get_context()

        success, msg = self._validate_mdc_config()

        if not success:
            return self._response(correlation_context, False, msg)

        config = self.config

        percentage = config.collection_sample_rate_percentage(self.name)

        if percentage < 100:
            if percentage <= random.random() * 100.0:
                # expected drop
                self.logger.debug("sampling not hit, drop the data")
                # TBD: send empty message to mdc to collect metrics of dropped messages?
                return self._response(correlation_context, False, "dropped_sampling")

        try:
            # build payload and put into payload queue
            log_data = _build_log_data_by_type(data)
        except TypeError as e:
            # unexpected drop
            self.on_error(e)
            return self._response(correlation_context, False, e.args[0])

        payload = build_payload(
            self.name,
            data=log_data,
            model_version=config.model_version(),
            context=correlation_context)

        success, msg = get_queue().enqueue(payload)

        if not success:
            # unexpected drop
            self.on_error(Exception(msg))
        return self._response(correlation_context, success, msg)

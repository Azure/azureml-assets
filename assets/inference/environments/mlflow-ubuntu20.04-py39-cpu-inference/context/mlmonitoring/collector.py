"""For collector."""
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Callable, Any

from .collector_json import JsonCollector
from .context import CorrelationContext


class Collector:
    """For collector class."""

    def __init__(
            self,
            *,
            name: str,
            on_error: Callable[[Exception], Any] = None
    ):
        """For init."""
        self._impl = JsonCollector(name=name, on_error=on_error)

    def collect(
            self,
            data,  # supported type: Union[pd.DataFrame]
            correlation_context: CorrelationContext = None) -> CorrelationContext:
        """For collect."""
        return self._impl.collect(data, correlation_context)

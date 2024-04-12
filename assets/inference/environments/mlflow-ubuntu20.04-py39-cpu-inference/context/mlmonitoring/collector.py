# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
from typing import Callable, Any

from .collector_json import JsonCollector
from .context import CorrelationContext


class Collector:
    def __init__(
            self,
            *,
            name: str,
            on_error: Callable[[Exception], Any] = None
    ):
        self._impl = JsonCollector(name=name, on_error=on_error)

    def collect(
            self,
            data,  # supported type: Union[pd.DataFrame]
            correlation_context: CorrelationContext = None) -> CorrelationContext:
        return self._impl.collect(data, correlation_context)

"""For Collector base."""
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import logging

from .init import init, is_sdk_ready
from .config import get_config
from .context import CorrelationContext, get_context_wrapper


class CollectorBase:
    """For CollectorBase."""

    def __init__(
            self,
            model_version: str):
        """For init."""
        if not is_sdk_ready():
            init(model_version)

        self.logger = logging.getLogger("mdc.collector")
        self.config = get_config()

    def _response(
            self,
            context: CorrelationContext,
            success: bool,
            message: str) -> CorrelationContext:
        """For response."""
        if self.config.is_debug():
            return get_context_wrapper(context, success, message)

        return context

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Cancel handler class."""

import signal
from proxy_component import AzureOpenAIProxyComponent
from common.logging import get_logger, add_custom_dimenions_to_app_insights_handler
logger = get_logger(__name__)


class CancelHandler:
    """Cancel handler."""
    def __init__(self, component: AzureOpenAIProxyComponent):
        """Create CancelHandler class."""
        self.component = component
        signal.signal(signal.SIGINT, self.cancel)
        signal.signal(signal.SIGTERM, self.cancel)

    def cancel(self, *args):
        """Cancel method to handle the signal."""
        if self.component is not None:
            logger.info("calling cancel job for the component")
            self.component.cancel_job()
        else:
            logger.warning("component is none for the cancel handler")

    def register_cancel_handler(component: AzureOpenAIProxyComponent):
        add_custom_dimenions_to_app_insights_handler(logger, component)
        logger.info("registering cancel handler")
        return CancelHandler(component)

    

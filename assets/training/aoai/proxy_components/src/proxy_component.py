# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Proxy component base class constructor."""


class AzureOpenAIProxyComponent:
    """Proxy component base class."""

    def __init__(self, endpoint_name, endpoint_resource_group, endpoint_subscription):
        """Proxy component base class constructor."""
        self.endpoint_name = endpoint_name
        self.endpoint_resource_group = endpoint_resource_group
        self.endpoint_subscription = endpoint_subscription

    def cancel_job(self):
        """Cancel method to handle the signal."""
        pass  # No-op function, does nothing
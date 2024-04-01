# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Proxy component base class."""


class AzureOpenAIProxyComponent:
    def __init__(self, endpoint_name, endpoint_resource_group, endpoint_subscription):
        self.endpoint_name = endpoint_name
        self.endpoint_resource_group = endpoint_resource_group
        self.endpoint_subscription = endpoint_subscription

    def cancel_job(self):
        pass  # No-op function, does nothing

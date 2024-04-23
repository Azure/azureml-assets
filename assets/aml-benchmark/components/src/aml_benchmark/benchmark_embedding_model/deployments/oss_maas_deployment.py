# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""OSS MaaS Deployment."""

from .oss_deployment import OSSDeployment


class OSSMaaSDeployment(OSSDeployment):
    """Class for OSS MaaS Deployment."""

    def __init__(
        self,
        endpoint_url: str,
        api_key: str,
    ):
        """Initialize Deployment."""
        if not endpoint_url.endswith("/v1/embed"):
            endpoint_url = endpoint_url + "/v1/embed"
        super().__init__(
            endpoint_url=endpoint_url,
            api_key=api_key,
        )

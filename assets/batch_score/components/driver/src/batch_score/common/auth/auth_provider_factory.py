# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains the definition for auth provider factory."""

from ..configuration.configuration_parser import Configuration
from .auth_provider import (
    ApiKeyAuthProvider,
    AuthProvider,
    IdentityAuthProvider,
    WorkspaceConnectionAuthProvider
)


class AuthProviderFactory:
    """Defines the auth provider factory."""

    def get_auth_provider(self, configuration: Configuration) -> AuthProvider:
        """Gets an instance of auth provider based on the configuration."""
        if configuration.authentication_type == "api_key":
            return ApiKeyAuthProvider(configuration.api_key_name)
        elif configuration.authentication_type == "managed_identity":
            return IdentityAuthProvider(use_user_identity=False)
        elif configuration.authentication_type in ["azureml_workspace_connection", "connection"]:
            endpoint_type = configuration.get_endpoint_type()
            return WorkspaceConnectionAuthProvider(configuration.connection_name, endpoint_type)
        else:
            raise Exception(f"Invalid authentication type {configuration.authentication_type}")

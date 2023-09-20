# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""The class for token provider."""

from typing import Dict
import os
from datetime import datetime, timezone
from azure.identity import ManagedIdentityCredential, CredentialUnavailableError, DefaultAzureCredential
from azure.core.credentials import AccessToken
from . import logging_utils as lu


class TokenProvider:
    """Class for token provider."""

    SCOPE_AML = "https://ml.azure.com/.default"
    SCOPE_ARM = "https://management.azure.com/.default"

    def __init__(self, client_id: str = None, token_file_path: str = None) -> None:
        """Init method."""
        if client_id is None:
            client_id = os.environ.get('DEFAULT_IDENTITY_CLIENT_ID')

        self.__credential: ManagedIdentityCredential = None
        self.__msi_access_tokens: Dict[str, AccessToken] = {}
        if client_id is not None:
            self.__credential = ManagedIdentityCredential(client_id=client_id)
        elif not token_file_path:
            self.__credential = DefaultAzureCredential()

        self.__token_file_path: str = token_file_path
        self.__file_token: str = None

    def get_token(self, scope: str) -> str:
        """Get token."""
        if self.__credential is not None:
            return self.__get_msi_access_token(scope).token

        return self.__get_token_from_file()

    def __is_msi_access_token_expired(self, scope: str) -> bool:
        return not self.__msi_access_tokens.get(scope) or \
            self.__msi_access_tokens[scope].expires_on <= datetime.now(timezone.utc).timestamp() + (5 * 60)

    def __get_msi_access_token(self, scope: str) -> AccessToken:
        # If there's a token that isn't expired, return that.
        if not self.__is_msi_access_token_expired(scope):
            return self.__msi_access_tokens[scope]

        try:
            lu.get_logger().debug("Attempting to get token from MSI")
            # Get token
            self.__msi_access_tokens[scope] = self.__credential.get_token(scope)
        except CredentialUnavailableError as e:
            print(e)
            lu.get_logger().error("Failed to get token from MSI")

        return self.__msi_access_tokens[scope]

    def __get_token_from_file(self):
        if self.__file_token is not None:
            return self.__file_token

        try:
            lu.get_logger().debug("Attempting to get token from file")

            # if file doesnt exist or value is none, read from env var
            if self.__token_file_path is None or not os.path.exists(self.__token_file_path):
                lu.get_logger().debug("Getting token from env var")
                self.__file_token = os.environ.get('TOKEN')
                return self.__file_token

            with open(self.__token_file_path, 'r') as file:
                self.__file_token = file.read().replace('\n', '')
        except Exception as e:
            print(e)
            lu.get_logger().error("Failed to get token from file")

        return self.__file_token

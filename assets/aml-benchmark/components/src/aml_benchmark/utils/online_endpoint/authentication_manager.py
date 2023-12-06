# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Class for credential used for the run."""

import os

from azureml.core import Run
from azure.identity import ManagedIdentityCredential

from ..logging import get_logger


logger = get_logger(__name__)


class AuthenticationManager:
    """Class for authentication related information used for the run."""
    ENV_CLIENT_ID_KEY = "DEFAULT_IDENTITY_CLIENT_ID"

    def __init__(self) -> None:
        if self._get_client_id() is None:
            logger.info("Client id is not provided.")
            self._credential = None
        else:
            self._credential = self._get_credential()

    def _get_client_id(self) -> str:
        """Get the client id."""
        return os.environ.get(AuthenticationManager.ENV_CLIENT_ID_KEY, None)

    def _get_credential(self) -> ManagedIdentityCredential:
        """Get the credential."""
        credential = ManagedIdentityCredential(
            client_id=self._get_client_id())
        return credential

    @property
    def credential(self) -> ManagedIdentityCredential:
        """Get the credential."""
        return self._credential

    @property
    def curr_workspace_bearer_token(self) -> str:
        """Get the workspace bearer token."""
        ws = Run.get_context().experiment.workspace
        if hasattr(ws._auth, "get_token"):
            bearer_token = ws._auth.get_token(
                "https://management.azure.com/.default").token
        else:
            bearer_token = ws._auth.token
        return bearer_token

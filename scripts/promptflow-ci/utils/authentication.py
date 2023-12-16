# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Auth utils."""

import uuid

from azureml.core.authentication import AzureCliAuthentication


def get_azure_cli_authentication_header(request_id=None):
    """Get login auth header."""
    interactive_auth = AzureCliAuthentication()
    header = interactive_auth.get_authentication_header()
    if request_id is None:
        request_id = str(uuid.uuid4())
    # add request id to header for tracking
    header["x-ms-client-request-id"] = request_id
    header["Content-Type"] = "application/json"
    header["Accept"] = "application/json"

    return header

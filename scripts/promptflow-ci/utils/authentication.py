# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import uuid

from azureml.core.authentication import InteractiveLoginAuthentication
from azureml.core.authentication import ServicePrincipalAuthentication


def get_service_principal_authentication_header(tenant_id, client_id, client_secret, request_id=None):
    """Get SP auth header"""
    auth = ServicePrincipalAuthentication(
        tenant_id=tenant_id,
        service_principal_id=client_id,
        service_principal_password=client_secret)
    header = auth.get_authentication_header()
    if request_id is None:
        request_id = str(uuid.uuid4())
    # add request id to header for tracking
    header["x-ms-client-request-id"] = request_id
    header["Content-Type"] = "application/json"
    header["Accept"] = "application/json"

    return header


def get_interactive_login_authentication_header(request_id=None):
    """Get login auth header"""
    interactive_auth = InteractiveLoginAuthentication()
    header = interactive_auth.get_authentication_header()
    if request_id is None:
        request_id = str(uuid.uuid4())
    # add request id to header for tracking
    header["x-ms-client-request-id"] = request_id
    header["Content-Type"] = "application/json"
    header["Accept"] = "application/json"

    return header

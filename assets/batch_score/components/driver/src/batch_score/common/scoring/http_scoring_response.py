# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains the data class for HTTP scoring response."""

from dataclasses import dataclass, field


@dataclass
class HttpScoringResponse:
    """Http scoring response."""

    headers: dict = field(default_factory=dict)
    payload: any = None
    status: int = None

    exception: Exception = None
    exception_traceback: str = None
    exception_type: str = None

    def get_model_response_code(self):
        """Get model Http response code."""
        return self.headers.get("ms-azureml-model-error-statuscode")

    def get_model_response_reason(self):
        """Get model Http response reason."""
        return self.headers.get("ms-azureml-model-error-reason")

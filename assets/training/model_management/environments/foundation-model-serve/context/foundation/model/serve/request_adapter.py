# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Request adapter module for input request transformation and validation.

This module provides adapters for transforming and validating incoming requests
before forwarding them to the inference engine.
"""
import copy
from fastapi import HTTPException
from foundation.model.serve.constants import ExtraParameters


def get_adapter(request, raw_request):
    """Get the appropriate adapter for the request.
    
    Args:
        request: The parsed request object.
        raw_request: The raw FastAPI request object.
        
    Returns:
        BaseAdapter: An adapter instance for the request.
    """
    # Default return original request object
    return BaseAdapter(request, raw_request)


class BaseAdapter:
    """Default adapter for input request transformation and validation."""

    def __init__(self, req, raw_req):
        """Initialize the BaseAdapter with the given request.
        
        Args:
            req: The parsed request object.
            raw_req: The raw FastAPI request object.
        """
        self.req = copy.deepcopy(req)
        self.headers = copy.deepcopy(raw_req.headers)
        self.path = getattr(raw_req.url, "path", "")

    def adapt(self):
        """Adapt and validate the request.
        
        Returns:
            The adapted request object.
        """
        self.validate()
        return self.req

    def validate(self):
        """Validate input request for all models.
        
        Raises:
            HTTPException: If validation fails.
        """
        # Skip validation if path is /v1/completions or /v1/chat/completions

        if self.path in ["/completions", "/chat/completions"]:
            extra_params_setting_str = self.headers.get(
                ExtraParameters.KEY, None)
            if extra_params_setting_str is not None and extra_params_setting_str not in ExtraParameters.OPTIONS:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unexpected EXTRA_PARAMETERS option {extra_params_setting_str}, "
                    f"expected options are {ExtraParameters.OPTIONS}"
                )

            # default to be error
            if (extra_params_setting_str is None or extra_params_setting_str == "error") \
                    and self.req.model_extra and len(self.req.model_extra) > 0:
                raise HTTPException(
                    status_code=400,
                    detail=f"Extra parameters {list(self.req.model_extra.keys())} are not allowed "
                    f"when {ExtraParameters.KEY} is not set or set to be '{ExtraParameters.ERROR}'. "
                    f"Set extra-parameters to '{ExtraParameters.PASS_THROUGH}' to pass to the model."
                )

            self.req._extra_param_setting = extra_params_setting_str

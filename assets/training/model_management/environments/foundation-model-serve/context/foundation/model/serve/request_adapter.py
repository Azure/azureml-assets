# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Adapter for input request."""
import copy
from fastapi import HTTPException
from foundation.model.serve.constants import ExtraParameters

def get_adapter(request, raw_request):
    # Default return original request object
    return BaseAdapter(request, raw_request)


class BaseAdapter:
    """Default adapter for input request."""

    def __init__(self, req, raw_req):
        """Initialize the BaseAdapter with the given request."""
        self.req = copy.deepcopy(req)
        self.headers = copy.deepcopy(raw_req.headers)
        self.path = getattr(raw_req.url, "path", "")

    def adapt(self):
        """Return the original request."""
        self.validate()
        return self.req

    def validate(self):
        """Validate input request for all the models."""
        # Skip validation if path is /v1/completions or /v1/chat/completions

        if self.path in ["/completions", "/chat/completions"]:
            extra_params_setting_str = self.headers.get(ExtraParameters.KEY, None)
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
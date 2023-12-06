# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from .dv3_estimator import DV3Estimator


# Estimator for the /completions API
class CompletionEstimator(DV3Estimator):
    def _get_prompt(self, request_obj: any) -> str:
        prompt = request_obj.get("prompt", None)

        if not prompt:
            raise Exception("Unsupported model input payload: cannot determine prompt for request cost estimation.")

        return prompt

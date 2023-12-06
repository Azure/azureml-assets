# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from ...telemetry import logging_utils as lu
from .request_modifier import RequestModifier


class PromptRedactionModifier(RequestModifier):
    def modify(self, request_obj: any) -> any:
        # See https://platform.openai.com/docs/api-reference
        fields_to_redact = [
            'input',  # /embeddings
            'messages',  # /chat-completions
            'prompt',  # /completions
            'transcript',  # /rainbow (Vesta)
        ]
        for field in fields_to_redact:
            if field in request_obj:
                request_obj[field] = 'REDACTED'
                lu.get_logger().debug("Redacted {} from request.".format(field))

        return request_obj

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Prompt redaction modifier."""

from .request_modifier import RequestModifier


class PromptRedactionModifier(RequestModifier):
    """Prompt redaction modifier."""

    def modify(self, request_obj: any) -> any:
        """Modify the request object."""
        # See https://platform.openai.com/docs/api-reference
        fields_to_redact = [
            'messages',  # /chat-completions
            'prompt',  # /completions
        ]
        for field in fields_to_redact:
            if field in request_obj:
                request_obj[field] = 'REDACTED'

        return request_obj

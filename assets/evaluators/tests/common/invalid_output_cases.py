# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Shared expectations for the privacy-safe ``_invalid_output_message`` helper in builtin evaluators."""

SECRET = "CUSTOMER-SECRET-TEXT"

INVALID_OUTPUT_CASES = [
    (None, "returned an empty response", "finish_reason=unknown, output_tokens=unknown"),
    ({"llm_output": "", "finish_reason": "stop", "output_token_count": 0}, "returned an empty response",
     "finish_reason=stop, output_tokens=0"),
    ({"llm_output": f'{{"score": 4, "reason": "{SECRET}', "finish_reason": "length", "output_token_count": 800},
     "output was truncated", "finish_reason=length, output_tokens=800"),
    ({"llm_output": SECRET, "finish_reason": "content_filter", "output_token_count": 3},
     "blocked by the content filter", "finish_reason=content_filter"),
    ({"llm_output": SECRET, "finish_reason": "stop", "output_token_count": 5}, "was not a valid JSON object",
     "finish_reason=stop, output_tokens=5"),
    ({"llm_output": [SECRET], "finish_reason": "stop"}, "had unexpected type list", "finish_reason=stop"),
    # Raw non-dict flow output is diagnosed by its own type, not reported as empty.
    (SECRET, "was not a valid JSON object", "finish_reason=unknown, output_tokens=unknown"),
    ([SECRET], "had unexpected type list", "finish_reason=unknown"),
    # Unexpected metadata values are never echoed.
    ({"llm_output": "", "finish_reason": SECRET, "output_token_count": {"content": SECRET}},
     "returned an empty response", "finish_reason=unknown, output_tokens=unknown"),
    # The _do_eval NaN check passes the result properties, which carry the token metadata.
    ({"reason": SECRET, "finish_reason": "stop", "completion_tokens": 12}, "no valid score could be parsed",
     "finish_reason=stop, output_tokens=12"),
]


def assert_invalid_output_diagnostics(invalid_output_message):
    """Assert the helper names the failure cause and safe metadata without echoing judge output."""
    for model_output, cause, details in INVALID_OUTPUT_CASES:
        message = invalid_output_message(model_output)
        assert message.startswith("Evaluator returned invalid output: "), message
        assert cause in message, message
        assert details in message, message
        assert SECRET not in message, message

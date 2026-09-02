# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Capture cached prompt-token usage from Prompty's OpenAI calls."""

import contextvars
import functools
from typing import Any, Optional


_CACHED_TOKENS: contextvars.ContextVar[Optional[int]] = contextvars.ContextVar(
    "evaluator_cached_tokens", default=None
)
_CAPTURE_MARKER = "_azureml_assets_captures_cached_tokens"


def extract_cached_tokens(response: Any) -> Optional[int]:
    """Read ``usage.prompt_tokens_details.cached_tokens`` from a response."""
    usage = response.get("usage") if isinstance(response, dict) else getattr(response, "usage", None)
    if usage is None:
        return None
    details = (
        usage.get("prompt_tokens_details")
        if isinstance(usage, dict)
        else getattr(usage, "prompt_tokens_details", None)
    )
    if details is None:
        return None
    cached_tokens = (
        details.get("cached_tokens")
        if isinstance(details, dict)
        else getattr(details, "cached_tokens", None)
    )
    return cached_tokens if isinstance(cached_tokens, int) and not isinstance(cached_tokens, bool) else None


def install_cached_token_capture() -> None:
    """Install the shared idempotent wrapper used by all Prompty evaluators."""
    try:
        from openai.resources.chat.completions import AsyncCompletions
    except ImportError:
        return

    original_create = getattr(AsyncCompletions, "create", None)
    if original_create is None or getattr(original_create, _CAPTURE_MARKER, False):
        return

    @functools.wraps(original_create)
    async def create(self, *args, **kwargs):
        response = await original_create(self, *args, **kwargs)
        cached_tokens = extract_cached_tokens(response)
        if cached_tokens is not None:
            _CACHED_TOKENS.set(cached_tokens)
        return response

    setattr(create, _CAPTURE_MARKER, True)
    AsyncCompletions.create = create


def clear_cached_tokens() -> None:
    """Clear cached-token usage before starting an evaluator call."""
    _CACHED_TOKENS.set(None)


def get_cached_tokens() -> Optional[int]:
    """Return cached-token usage captured in the current async context."""
    return _CACHED_TOKENS.get()

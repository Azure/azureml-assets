# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from ...common.auth.token_provider import TokenProvider
from .open_ai_header_handler import OpenAIHeaderHandler


class VestaChatCompletionHeaderHandler(OpenAIHeaderHandler):
    def __init__(
            self,
            token_provider: TokenProvider,
            component_version: str = None,
            user_agent_segment: str = None,
            batch_pool: str = None,
            quota_audience: str = None,
            additional_headers: str = None) -> None:
        super().__init__(
            token_provider,
            component_version,
            user_agent_segment,
            batch_pool,
            quota_audience,
            additional_headers)

        self._additional_headers.setdefault(
            "Openai-Internal-HarmonyVersion",
            "harmony_v4.0.11_8k_turbo_mm")

        self._additional_headers.setdefault(
            "Openai-Internal-AllowedSpecialTokens",
            "<|im_start|>,<|im_sep|>,<|im_end|>,<|diff_marker|>,<|fim_suffix|>,"
            "<|ghreview|>,<|ipynb_marker|>,<|fim_middle|>,<|meta_start|>")

        self._additional_headers.setdefault(
            "Openai-Internal-AllowedOutputSpecialTokens",
            "<|im_start|>,<|im_sep|>,<|im_end|>,<|diff_marker|>,<|fim_suffix|>,"
            "<|ghreview|>,<|ipynb_marker|>,<|fim_middle|>,<|meta_start|>")

        self._additional_headers.setdefault(
            "Openai-Internal-MegatokenSwitchAndParams",
            "true-true-1-4-2000")

from ...common.auth.token_provider import TokenProvider
from .open_ai_header_handler import OpenAIHeaderHandler


class ChatCompletionHeaderHandler(OpenAIHeaderHandler):
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
            "Openai-Internal-AllowedSpecialTokens", "<|im_start|>,<|im_sep|>,<|im_end|>")

        self._additional_headers.setdefault(
            "Openai-Internal-AllowedOutputSpecialTokens", "<|im_start|>,<|im_sep|>,<|im_end|>")

        self._additional_headers.setdefault(
            "Openai-Internal-HarmonyVersion", "harmony_v3")

        self._additional_headers.setdefault(
            "Openai-Internal-AllowChatCompletion", "true")

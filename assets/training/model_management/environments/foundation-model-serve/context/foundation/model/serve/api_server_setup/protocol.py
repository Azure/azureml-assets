import json
import re
import time
from enum import Enum
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Union,
)

from jsonschema import (
    Draft7Validator,
    SchemaError,
)
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

import uuid


class ChatRole(str, Enum):
    system = "system"
    user = "user"
    assistant = "assistant"
    tool = "tool"


class ContentImageDetail(str, Enum):
    auto = "auto"
    low = "low"
    high = "high"


class ContentPartType(str, Enum):
    text = "text"
    image = "image"
    image_url = "image_url"
    audio_url = "audio_url"
    _input_audio = "input_audio"


class ToolType(str, Enum):
    function = "function"


class Function(BaseModel):
    name: str
    arguments: Union[Dict, str]
    call_id: Optional[str] = None

    @field_validator("arguments")
    def cast_to_dict(cls, v):
        if isinstance(v, str):
            try:
                loaded = json.loads(v)
                if isinstance(loaded, dict):
                    return loaded
                else:
                    return v
            except json.JSONDecodeError:
                return v
        return v

    @field_validator("name")
    def must_be_alphanumeric_or_underscore_dash(cls, v: str) -> str:
        if not re.match(r"^[a-zA-Z0-9_-]{1,64}$", v):
            raise ValueError(
                f"Function name was {v} but must be a-z, A-Z, 0-9, "
                "or contain underscores and dashes, with a maximum length of 64."
            )
        return v

    model_config = ConfigDict(
        extra="allow",
        exclude_unset=True,
    )


class ChatCompletionMessageToolCall(BaseModel):
    id: Optional[str] = None
    type: ToolType = ToolType.function
    function: Function

    model_config = ConfigDict(
        extra="allow",
        exclude_unset=True,
    )


class ContentImage(BaseModel):
    data: Any
    detail: Optional[ContentImageDetail] = ContentImageDetail.auto

    model_config = ConfigDict(
        extra="allow",
        exclude_unset=True,
    )


class ContentImageUrl(BaseModel):
    url: str
    detail: Optional[ContentImageDetail] = ContentImageDetail.auto

    model_config = ConfigDict(
        extra="allow",
        exclude_unset=True,
    )


class ContentAudioUrl(BaseModel):
    url: str

    model_config = ConfigDict(
        extra="allow",
        exclude_unset=True,
    )


class ContentAudioFormat(str, Enum):
    wav = "wav"
    mp3 = "mp3"


class ContentInputAudio(BaseModel):
    data: Any
    format: ContentAudioFormat = ContentAudioFormat.wav

    model_config = ConfigDict(
        extra="allow",
        exclude_unset=True,
    )


class ContentPart(BaseModel):
    type: ContentPartType
    text: Optional[str] = None
    image: Optional[ContentImage] = None
    image_url: Optional[ContentImageUrl] = None
    audio_url: Optional[ContentAudioUrl] = None
    _input_audio: Optional[ContentInputAudio] = None

    model_config = ConfigDict(
        extra="allow",
        exclude_unset=True,
    )

    @model_validator(mode="after")
    def validate_content_part(self):
        if self.type == ContentPartType.text and not self.text:
            raise ValueError(f"Content part type is {self.type} but text is not provided.")
        if self.type == ContentPartType.image and not self.image:
            raise ValueError(f"Content part type is {self.type} but image is not provided.")
        if self.type == ContentPartType.image_url and not self.image_url:
            raise ValueError(f"Content part type is {self.type} but image_url is not provided.")
        if self.type == ContentPartType.audio_url and not self.audio_url:
            raise ValueError(f"Content part type is {self.type} but audio_url is not provided.")
        if self.type == ContentPartType._input_audio and not self._input_audio:
            raise ValueError(f"Content part type is {self.type} but input_audio is not provided.")
        return self


class ChatMessage(BaseModel):
    role: ChatRole
    content: Optional[Union[str, List[ContentPart]]] = None
    tool_calls: Optional[List[ChatCompletionMessageToolCall]] = None

    model_config = ConfigDict(
        extra="allow",
        exclude_unset=True,
    )

    def __init__(self, **data):
        super().__init__(**data)
        if self.tool_calls:
            for (
                index,
                tool_call,
            ) in enumerate(self.tool_calls):
                if tool_call.id is None:
                    tool_call.id = f"call_{tool_call.function.name}_{index}"


class ResponseFormat(BaseModel):
    type: str

    @field_validator("type")
    def must_be_valid_response_format(cls, v: str) -> str:
        if v.lower() not in [
            "text",
            "json_object",
        ]:
            raise ValueError(f"Response format was {v} but must be either 'text' or 'json_object'.")
        return v


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: Optional[Any] = None

    @field_validator("parameters")
    def validate_parameters(cls, v: dict) -> dict:
        try:
            Draft7Validator.check_schema(v)
        except SchemaError as e:
            raise ValueError(e.message)
        return v

    @field_validator("name")
    def must_be_alphanumeric_or_underscore_dash(cls, v: str) -> str:
        if not re.match(r"^[a-zA-Z0-9_-]{1,64}$", v):
            raise ValueError(
                f"Function name was {v} but must be a-z, A-Z, 0-9, "
                "or contain underscores and dashes, with a maximum length of 64."
            )
        return v

    model_config = ConfigDict(
        extra="allow",
        exclude_unset=True,
    )


class ChatCompletionToolParam(BaseModel):
    type: ToolType = ToolType.function
    function: FunctionDefinition = None


class BaseRequest(BaseModel):
    model_config = ConfigDict(
        extra="allow",
        exclude_unset=True,
    )

    model: Optional[str] = Field(
        None,
        description=("The model to use for completion,"
                     "this will have no effect on an endpoint deployed for a specific model."),
    )
    frequency_penalty: float = Field(
        0.0,
        ge=-2.0,
        le=2.0,
        description=("A value that influences the probability of generated tokens appearing"
                     "based on their cumulative frequency in generated text. Has a valid range of -2.0 to 2.0"),
    )
    presence_penalty: float = Field(
        0.0,
        ge=-2.0,
        le=2.0,
        description=("A value that influences the probability of generated tokens appearing "
                     "based on their existing presence in the generated text. Has a valid range of -2.0 to 2.0"),
    )
    seed: Optional[int] = Field(
        None,
        description=("An integer value, if specified, will make a best effort to sample deterministically,"
                     "such that repeated requests with the same seed and parameters should return the same result."),
    )
    temperature: float = Field(
        0.7,
        ge=0.0,
        le=1.0,
        description=("A non-negative float that tunes the degree of randomness in generation."
                     "Lower temperatures mean less random generations,"
                     "and higher temperatures mean more random generations."
                     "It is generally recommended to alter this or top_p but not both."),
    )
    top_p: float = Field(
        1.0,
        ge=0.0,
        le=1.0,
        description=("Nucleus sampling, where the model considers"
                     "the results of the tokens with top_p probability mass."
                     "So 0.1 means only the tokens comprising the top 10% probability mass are considered."
                     "It is generally recommended to alter this or temperature but not both."),
    )
    max_tokens: Optional[int] = Field(default=256, ge=0)
    stop: Optional[Union[str, List[str]]] = Field(
        default_factory=list,
        json_schema_extra={"redact": True},
    )
    stream: Optional[bool] = False

    def __repr__(self) -> str:
        model = self.model_dump()
        repr_str = "BaseRequest("
        for k, v in model.items():
            field = self.model_fields.get(k)
            if field and field.json_schema_extra and field.json_schema_extra.get("redact"):
                v = v
            repr_str += f"{k}={v}, "
        repr_str += ")"
        return repr_str

    def to_downstream_json(self, include_extra=False, logger=None) -> Dict[str, Any]:
        model = self.model_dump(exclude_unset=True)

        if include_extra is False:
            for k in self.model_extra:
                if k in model:
                    if logger:
                        logger.warning(f"[to_downstream_json] Removing extra key: {k}")
                    del model[k]

        return model


class CompletionRequest(BaseRequest):
    model_config = ConfigDict(
        extra="allow",
        exclude_unset=True,
    )

    prompt: str = Field(
        ...,
        description="Input prompt used for text generation",
    )

    def __repr__(self) -> str:
        model = self.model_dump()
        repr_str = "TextGenerationRequest("
        for k, v in model.items():
            field = self.model_fields.get(k)
            if field and field.json_schema_extra and field.json_schema_extra.get("redact"):
                v = v
            repr_str += f"{k}={v}, "
        repr_str += ")"
        return repr_str


class ChatCompletionRequest(BaseRequest):
    model_config = ConfigDict(
        extra="allow",
        exclude_unset=True,
    )

    messages: List[ChatMessage] = Field(
        ...,
        description="List of messages to be used for completion",
        json_schema_extra={"redact": True},
    )
    response_format: Optional[ResponseFormat] = Field(
        None,
        description="The format of the response, either 'text' or 'json_object'.",
    )
    tools: Optional[List[ChatCompletionToolParam]] = Field(
        None,
        json_schema_extra={"redact": True},
    )
    tool_choice: Optional[str] = None

    @field_validator("messages")
    def validate_messages_not_empty(cls, msgs):
        if len(msgs) == 0:
            raise ValueError("messages can not be an empty list")
        return msgs

    def __repr__(self) -> str:
        model = self.model_dump()
        repr_str = "ChatCompletionRequest("
        for k, v in model.items():
            field = self.model_fields.get(k)
            if field and field.json_schema_extra and field.json_schema_extra.get("redact"):
                v = v
            repr_str += f"{k}={v}, "
        repr_str += ")"
        return repr_str


class ChatCompletionRequestFreeFlow(BaseRequest):
    model_config = ConfigDict(
        extra="allow",         # accept extra params
        exclude_unset=True,    # drop unset values
    )

    messages: List["ChatMessage"] = Field(
        ...,
        description="List of messages to be used for completion",
        json_schema_extra={"redact": True},
    )
    response_format: Optional["ResponseFormat"] = Field(
        None,
        description="The format of the response, either 'text' or 'json_object'.",
    )
    tools: Optional[List["ChatCompletionToolParam"]] = Field(
        None,
        json_schema_extra={"redact": True},
    )
    tool_choice: Optional[str] = None

    # <--- new: all unknown params collected here
    extra_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Holds any extra parameters not explicitly defined in the schema.",
    )

    @field_validator("messages")
    def validate_messages_not_empty(cls, msgs):
        if len(msgs) == 0:
            raise ValueError("messages can not be an empty list")
        return msgs

    @model_validator(mode="before")
    def collect_extra_params(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Move unknown keys into extra_params"""
        known_fields = set(cls.model_fields.keys())
        extras = {k: v for k, v in values.items() if k not in known_fields}
        if extras:
            values["extra_params"] = extras
        return values

    def __repr__(self) -> str:
        model = self.model_dump()
        repr_str = "ChatCompletionRequest("
        for k, v in model.items():
            field = self.model_fields.get(k)
            if (
                field
                and field.json_schema_extra
                and field.json_schema_extra.get("redact")
            ):
                v = v
            repr_str += f"{k}={v}, "
        repr_str += ")"
        return repr_str


# Response
class FinishReason(str, Enum):
    stop: str = "stop"
    length: str = "length"
    model_length: str = "model_length"
    error: str = "error"
    tool_call: str = "tool_calls"
    content_filter: str = "content_filter"


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[FinishReason] = None

    model_config = ConfigDict(
        extra="ignore",
        exclude_unset=True,
    )


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0

    model_config = ConfigDict(
        extra="allow",
        exclude_unset=True,
    )


class ChatCompletionResponse(BaseModel):
    id: str  # internal X-Request-Id from request headers
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: UsageInfo

    model_config = ConfigDict(
        extra="ignore",
        exclude_unset=True,
    )


class ChoiceDeltaToolCall(BaseModel):
    id: str
    index: int
    type: str
    function: Function

    model_config = ConfigDict(
        extra="allow",
        exclude_unset=True,
    )


class DeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None
    tool_calls: Optional[List[ChoiceDeltaToolCall]] = None

    model_config = ConfigDict(extra="allow", exclude_unset=True)


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[FinishReason] = None

    model_config = ConfigDict(extra="ignore", exclude_unset=True)


class ChatCompletionStreamResponse(BaseModel):
    id: str  # internal X-Request-Id from request headers
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseStreamChoice]
    usage: Optional[UsageInfo] = Field(default=None)

    model_config = ConfigDict(
        extra="ignore",
        exclude_unset=True,
    )


class LogProbs(BaseModel):
    """Class that represents logprobs in the openai way."""

    text_offset: List[int] = Field(default_factory=list)
    token_logprobs: List[Optional[float]] = Field(default_factory=list)
    tokens: List[str] = Field(default_factory=list)
    top_logprobs: List[Optional[Dict[str, float]]] = Field(default_factory=list)


class CompletionResponseChoice(BaseModel):
    """Class for one of the 'choices' in the openai api text generation response."""

    index: int = 0
    text: str = ""
    logprobs: Optional[LogProbs] = None
    finish_reason: Optional[
        Literal[
            "stop",
            "length",
            "content_filter",
        ]
    ] = None

    model_config = ConfigDict(extra="ignore", exclude_unset=True)


class CompletionResponse(BaseModel):
    """An openai text generation response object."""

    id: str = Field(default_factory=lambda: f"cmpl-{uuid.uuid4()}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionResponseChoice]
    usage: UsageInfo

    model_config = ConfigDict(extra="ignore", exclude_unset=True)


class CompletionResponseStreamChoice(BaseModel):
    index: int = 0
    text: str = ""
    logprobs: Optional[LogProbs] = None
    finish_reason: Optional[Literal["stop", "length", "content_filter"]] = None

    model_config = ConfigDict(extra="ignore", exclude_unset=True)


class CompletionStreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{uuid.uuid4()}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionResponseStreamChoice]
    usage: Optional[UsageInfo] = Field(default=None)

    model_config = ConfigDict(extra="ignore", exclude_unset=True)


class AzureError(BaseModel):
    code: str
    message: str
    status: int


class AzureErrorResponse(BaseModel):
    error: AzureError

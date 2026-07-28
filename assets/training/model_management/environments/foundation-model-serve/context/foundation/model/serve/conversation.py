# Licensed under the MIT License.
# Copyright (c) Microsoft Corporation.
"""Module for handling conversation data, including text and multimodal messages."""
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any, Union


class Role(Enum):
    """Enum representing the role in a conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

    def __str__(self):
        """Return the string representation of the role."""
        return self.value


@dataclass
class Message:
    """Base class for chat messages."""

    role: Role

    def to_dict(self) -> dict:
        """Convert the message to a dictionary."""
        return {"role": self.role.value}


@dataclass
class TextMessage(Message):
    """Class representing text-based chat messages."""

    content: str

    def to_dict(self) -> dict:
        """Convert the text message to a dictionary."""
        base_dict = super().to_dict()
        base_dict["content"] = self.content
        return base_dict


@dataclass
class MultimodalContent:
    """Class for representing a multimodal content type."""

    type: str  # e.g., 'text', 'image_url', 'audio_url'
    text: Optional[str] = None
    image_url: Optional[Dict[str, str]] = None
    audio_url: Optional[Any] = None
    input_audio: Optional[Any] = None


@dataclass
class MultimodalMessage(Message):
    """Class representing multimodal chat messages."""

    content: List[MultimodalContent]

    def to_dict(self) -> dict:
        """Convert the multimodal message to a dictionary."""
        base_dict = super().to_dict()
        base_dict["content"] = [
            {k: v for k, v in vars(item).items() if v is not None} for item in self.content
        ]
        return base_dict


@dataclass
class Conversation:
    """Class representing a conversation."""

    messages: Optional[List[Union[TextMessage, MultimodalMessage]]] = field(
        default_factory=list)

    def __post_init__(self):
        """Validate the first message after initialization."""
        self.validate_first_message()

    def validate_first_message(self):
        """Validate that the first message is from the system or the user."""
        if self.messages:
            first_role = self.messages[0].role
            if first_role not in [Role.SYSTEM, Role.USER]:
                raise ValueError(
                    "The first message should be from the system or the user.")

    def add_message(self, message: Union[TextMessage, MultimodalMessage]):
        """Add a message to the conversation."""
        if len(self.messages) == 0 and message.role not in [Role.SYSTEM, Role.USER]:
            raise ValueError(
                "The first message should be from the system or the user.")
        self.messages.append(message)

    def to_json(self) -> str:
        """Serialize the conversation to a JSON string."""
        return json.dumps([msg.to_dict() for msg in self.messages], indent=4)

    @classmethod
    def from_json(cls, json_str: str):
        """Create a conversation from a JSON string."""
        msg_dicts = json.loads(json_str)
        messages = []
        for msg_dict in msg_dicts:
            if isinstance(msg_dict.get("content"), list):
                messages.append(MultimodalMessage(Role(msg_dict["role"]),
                                [MultimodalContent(**item) for item in msg_dict["content"]]))
            else:
                messages.append(TextMessage(
                    Role(msg_dict["role"]), msg_dict["content"]))
        return cls(messages)


if __name__ == "__main__":
    """Example usage of the Conversation class."""
    try:
        conv = Conversation(
            [TextMessage(Role.ASSISTANT, "Can't start with assistant.")])
    except ValueError as e:
        print(f"Validation Error: {e}")

    conv = Conversation(
        [TextMessage(Role.SYSTEM, "You are a helpful assistant.")])
    conv.add_message(TextMessage(
        Role.USER, "Who won the world series in 2020?"))
    conv.add_message(TextMessage(
        Role.ASSISTANT, "The Los Angeles Dodgers won the World Series in 2020."))
    conv.add_message(MultimodalMessage(Role.USER, [
        MultimodalContent(type="image_url", image_url={
                          "url": "https://example.com/image.jpg"}),
        MultimodalContent(type="audio_url", audio_url={
                          "url": "https://example.com/audio.wav"})
    ]))

    json_str = conv.to_json()
    print("Serialized to JSON:")
    print(json_str)

    conv = Conversation.from_json(json_str)
    print("Deserialized from JSON:")
    print(conv)

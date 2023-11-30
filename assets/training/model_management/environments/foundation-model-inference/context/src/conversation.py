# Licensed under the MIT License.
# Copyright (c) Microsoft Corporation.
"""Module for handling conversation data."""
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


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
    """Class representing a message in a conversation."""

    role: Role
    content: str

    def to_dict(self) -> dict:
        """Convert the message to a dictionary."""
        return {
            "role": self.role.value,
            "content": self.content,
        }

    @classmethod
    def from_dict(cls, msg_dict: dict):
        """Create a message from a dictionary."""
        role = Role(msg_dict["role"])
        content = msg_dict["content"]
        return cls(role, content)


@dataclass
class Conversation:
    """Class representing a conversation."""

    messages: Optional[List[Message]] = field(default_factory=list)

    def __post_init__(self):
        """Validate the first message after initialization."""
        self.validate_first_message()

    def validate_first_message(self):
        """Validate that the first message is from the system or the user."""
        if self.messages:
            first_role = self.messages[0].role
            if first_role not in [Role.SYSTEM, Role.USER]:
                raise ValueError(
                    "The first message should be from the system or the user.",
                )

    def add_message(self, message: Message):
        """Add a message to the conversation."""
        if len(self.messages) == 0:
            if message.role not in [Role.SYSTEM, Role.USER]:
                raise ValueError(
                    "The first message should be from the system or the user.",
                )
        self.messages.append(message)

    def to_json(self) -> str:
        """Serialize the conversation to a JSON string."""
        return json.dumps([msg.to_dict() for msg in self.messages], indent=4)

    @classmethod
    def from_json(cls, json_str: str):
        """Create a conversation from a JSON string."""
        msg_dicts = json.loads(json_str)
        messages = [Message.from_dict(msg_dict) for msg_dict in msg_dicts]
        return cls(messages)


if __name__ == "__main__":
    """Example usage of the Conversation class."""
    try:
        conv = Conversation([Message(Role.ASSISTANT, "Can't start with assistant.")])
    except ValueError as e:
        print(f"Validation Error: {e}")

    conv = Conversation([Message(Role.SYSTEM, "You are a helpful assistant.")])
    conv.add_message(Message(Role.USER, "Who won the world series in 2020?"))
    conv.add_message(
        Message(Role.ASSISTANT, "The Los Angeles Dodgers won the World Series in 2020."),
    )

    json_str = conv.to_json()
    print("Serialized to JSON:")
    print(json_str)

    conv = Conversation.from_json(json_str)
    print("Deserialized from JSON:")
    print(conv)

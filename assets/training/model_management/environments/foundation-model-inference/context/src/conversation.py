# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import dataclass, field
from typing import List, Union, Optional
from enum import Enum
import json


class Role(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

    def __str__(self):
        return self.value


@dataclass
class Message:
    role: Role
    content: str

    def to_dict(self) -> dict:
        return {
            "role": self.role.value,
            "content": self.content,
        }

    @classmethod
    def from_dict(cls, msg_dict: dict):
        role = Role(msg_dict["role"])
        content = msg_dict["content"]
        return cls(role, content)


@dataclass
class Conversation:
    messages: Optional[List[Message]] = field(default_factory=list)

    def __post_init__(self):
        self.validate_first_message()

    def validate_first_message(self):
        if self.messages:
            first_role = self.messages[0].role
            if first_role not in [Role.SYSTEM, Role.USER]:
                raise ValueError(
                    "The first message should be from the system or the user."
                )

    def add_message(self, message: Message):
        if len(self.messages) == 0:
            if message.role not in [Role.SYSTEM, Role.USER]:
                raise ValueError(
                    "The first message should be from the system or the user."
                )
        self.messages.append(message)

    def to_json(self) -> str:
        return json.dumps([msg.to_dict() for msg in self.messages], indent=4)

    @classmethod
    def from_json(cls, json_str: str):
        msg_dicts = json.loads(json_str)
        messages = [Message.from_dict(msg_dict) for msg_dict in msg_dicts]
        return cls(messages)


if __name__ == "__main__":
    # Example usage
    try:
        conv = Conversation([Message(Role.ASSISTANT, "Can't start with assistant.")])
    except ValueError as e:
        print(f"Validation Error: {e}")

    conv = Conversation([Message(Role.SYSTEM, "You are a helpful assistant.")])
    conv.add_message(Message(Role.USER, "Who won the world series in 2020?"))
    conv.add_message(
        Message(Role.ASSISTANT, "The Los Angeles Dodgers won the World Series in 2020.")
    )

    json_str = conv.to_json()
    print("Serialized to JSON:")
    print(json_str)

    conv = Conversation.from_json(json_str)
    print("Deserialized from JSON:")
    print(conv)

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""HMAC framing for vLLM rollout ZeroMQ RPC messages."""

import hashlib
import hmac


_SEQUENCE_SIZE = 8


def sign_message(key: bytes, sequence: int, payload: bytes, direction: str) -> list[bytes]:
    """Return ZeroMQ frames carrying the sequence, payload and HMAC signature."""
    sequence_frame = sequence.to_bytes(_SEQUENCE_SIZE, "big")
    signature = hmac.new(key, direction.encode("ascii") + b"\x00" + sequence_frame + payload, hashlib.sha256).digest()
    return [sequence_frame, payload, signature]


def verify_message(key: bytes, expected_sequence: int, frames: list[bytes], direction: str) -> bytes:
    """Validate signed ZeroMQ frames and return the payload, raising ValueError on failure."""
    if len(frames) != 3:
        raise ValueError("Invalid authenticated message frame count")

    sequence_frame, payload, signature = frames
    if len(sequence_frame) != _SEQUENCE_SIZE or int.from_bytes(sequence_frame, "big") != expected_sequence:
        raise ValueError("Invalid authenticated message sequence")

    expected_signature = hmac.new(
        key, direction.encode("ascii") + b"\x00" + sequence_frame + payload, hashlib.sha256
    ).digest()
    if not hmac.compare_digest(signature, expected_signature):
        raise ValueError("Invalid authenticated message signature")

    return payload

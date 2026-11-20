# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""HMAC framing for vLLM rollout ZeroMQ RPC messages."""

import hashlib
import hmac
import os
from collections.abc import Iterable


_SEQUENCE_SIZE = 8
_KEY_SIZE = 32


def validate_key(key: bytes) -> None:
    """Validate an HMAC-SHA256 authentication key."""
    if not isinstance(key, bytes):
        raise ValueError("ZeroMQ authentication key must be bytes")
    if len(key) != _KEY_SIZE:
        raise ValueError("ZeroMQ authentication key must be exactly 32 bytes")


def validate_ray_tls_for_addresses(addresses: Iterable[str]) -> None:
    """Require Ray TLS before authentication keys are used over TCP."""
    addresses = list(addresses)
    if not any(address.startswith("tcp://") for address in addresses):
        return

    if os.getenv("RAY_USE_TLS") != "1":
        raise RuntimeError(
            "TCP vLLM rollout RPC requires RAY_USE_TLS=1"
        )

    tls_names = (
        "RAY_TLS_SERVER_CERT",
        "RAY_TLS_SERVER_KEY",
        "RAY_TLS_CA_CERT",
    )
    for name in tls_names:
        path = os.getenv(name)
        if not path or not os.path.isfile(path):
            raise RuntimeError(
                f"TCP vLLM rollout RPC requires readable Ray TLS file {name}"
            )


def sign_message(
    key: bytes,
    sequence: int,
    payload: bytes,
    direction: str,
) -> list[bytes]:
    """Return ZeroMQ frames with the sequence, payload and signature."""
    validate_key(key)
    sequence_frame = sequence.to_bytes(_SEQUENCE_SIZE, "big")
    signature = hmac.new(
        key,
        direction.encode("ascii") + b"\x00" + sequence_frame + payload,
        hashlib.sha256,
    ).digest()
    return [sequence_frame, payload, signature]


def verify_message(
    key: bytes,
    expected_sequence: int,
    frames: list[bytes],
    direction: str,
) -> bytes:
    """Validate signed frames and return payload or raise ValueError."""
    validate_key(key)
    if len(frames) != 3:
        raise ValueError("Invalid authenticated message frame count")

    sequence_frame, payload, signature = frames
    if (
        len(sequence_frame) != _SEQUENCE_SIZE
        or int.from_bytes(sequence_frame, "big") != expected_sequence
    ):
        raise ValueError("Invalid authenticated message sequence")

    expected_signature = hmac.new(
        key,
        direction.encode("ascii") + b"\x00" + sequence_frame + payload,
        hashlib.sha256,
    ).digest()
    if not hmac.compare_digest(signature, expected_signature):
        raise ValueError("Invalid authenticated message signature")

    return payload

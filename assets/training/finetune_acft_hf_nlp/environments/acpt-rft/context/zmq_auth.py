# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""HMAC framing for vLLM rollout ZeroMQ RPC messages."""

import hashlib
import hmac
import os
import socket
import ssl
from collections.abc import Iterable


_SEQUENCE_SIZE = 8
_KEY_SIZE = 32


def validate_key(key: bytes) -> None:
    """Validate an HMAC-SHA256 authentication key."""
    if not isinstance(key, bytes):
        raise ValueError("ZeroMQ authentication key must be bytes")
    if len(key) != _KEY_SIZE:
        raise ValueError("ZeroMQ authentication key must be exactly 32 bytes")


def validate_ray_tls_for_addresses(
    addresses: Iterable[str],
) -> ssl.SSLContext | None:
    """Load verified Ray TLS credentials for TCP rollout endpoints."""
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

    try:
        context = ssl.create_default_context(
            cafile=os.environ["RAY_TLS_CA_CERT"]
        )
        context.load_cert_chain(
            os.environ["RAY_TLS_SERVER_CERT"],
            os.environ["RAY_TLS_SERVER_KEY"],
        )
    except (OSError, ssl.SSLError) as exc:
        raise RuntimeError("Invalid Ray TLS credentials") from exc
    return context


def _ray_worker_rpc_address() -> tuple[str, int]:
    # Ray 2.56 exposes the owning core worker address through object ownership.
    import ray
    from ray._private.worker import global_worker
    from ray.core.generated.common_pb2 import Address

    reference = ray.put(None)
    address = Address()
    address.ParseFromString(
        global_worker.core_worker.get_owner_address(reference)
    )
    if not address.ip_address or not address.port:
        raise RuntimeError("Cannot determine the Ray worker RPC address")
    return address.ip_address, address.port


def verify_ray_worker_tls(address: str) -> None:
    """Verify Ray's active TLS listener before releasing TCP credentials."""
    context = validate_ray_tls_for_addresses([address])
    if context is None:
        return
    host, port = _ray_worker_rpc_address()
    try:
        context.set_alpn_protocols(["h2"])
        with socket.create_connection((host, port), timeout=5) as connection:
            with context.wrap_socket(
                connection, server_hostname=host,
            ) as secure:
                # Complete an HTTP/2 exchange so TLS 1.3 client-auth alerts
                # cannot arrive after a successful client-side handshake.
                secure.sendall(
                    b"PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n"
                    b"\x00\x00\x00\x04\x00\x00\x00\x00\x00"
                )
                if not secure.recv(1):
                    raise OSError("Ray RPC TLS listener closed the connection")
    except (OSError, ssl.SSLError) as exc:
        raise RuntimeError(
            "Ray worker RPC must use verified TLS before key distribution"
        ) from exc


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

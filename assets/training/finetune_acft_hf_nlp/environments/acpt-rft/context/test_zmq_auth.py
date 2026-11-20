# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for vLLM rollout ZeroMQ message authentication."""

import tempfile
import unittest
from unittest.mock import patch

from zmq_auth import (
    sign_message,
    validate_ray_tls_for_addresses,
    verify_message,
)


class ZmqAuthTests(unittest.TestCase):
    """Validate signing and verification of rollout RPC frames."""

    def test_round_trip(self):
        """A signed message verifies and returns its payload."""
        key = b"k" * 32
        payload = b"serialized request"

        frames = sign_message(key, 0, payload, "request")
        self.assertEqual(
            verify_message(key, 0, frames, "request"),
            payload,
        )

    def test_rejects_tampered_payload(self):
        """A modified payload is rejected."""
        key = b"k" * 32
        frames = sign_message(key, 0, b"serialized request", "request")
        frames[1] = b"tampered request"

        with self.assertRaises(ValueError):
            verify_message(key, 0, frames, "request")

    def test_rejects_replayed_sequence(self):
        """A message with an unexpected sequence is rejected."""
        key = b"k" * 32
        frames = sign_message(key, 0, b"serialized request", "request")

        with self.assertRaises(ValueError):
            verify_message(key, 1, frames, "request")

    def test_rejects_invalid_frame_count(self):
        """A message with the wrong frame count is rejected."""
        with self.assertRaises(ValueError):
            verify_message(b"k" * 32, 0, [b"single frame"], "request")

    def test_rejects_wrong_direction(self):
        """A request signature cannot be reused as a response."""
        key = b"k" * 32
        frames = sign_message(key, 0, b"serialized request", "request")

        with self.assertRaises(ValueError):
            verify_message(key, 0, frames, "response")

    def test_rejects_wrong_key_length(self):
        """An authentication key must be exactly 32 bytes."""
        with self.assertRaises(ValueError):
            sign_message(b"too-short", 0, b"payload", "request")

    def test_tcp_requires_ray_tls(self):
        """TCP worker credentials require Ray TLS configuration."""
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "RAY_USE_TLS"):
                validate_ray_tls_for_addresses(
                    ["tcp://10.0.0.2:1234"]
                )

    def test_tcp_accepts_ray_tls_files(self):
        """TLS is accepted with Ray enabled and certificates present."""
        with (
            tempfile.NamedTemporaryFile() as cert,
            tempfile.NamedTemporaryFile() as key,
            tempfile.NamedTemporaryFile() as ca,
        ):
            env = {
                "RAY_USE_TLS": "1",
                "RAY_TLS_SERVER_CERT": cert.name,
                "RAY_TLS_SERVER_KEY": key.name,
                "RAY_TLS_CA_CERT": ca.name,
            }
            with patch.dict("os.environ", env, clear=True):
                validate_ray_tls_for_addresses(["tcp://10.0.0.2:1234"])


if __name__ == "__main__":
    unittest.main()

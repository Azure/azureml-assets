# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for vLLM rollout ZeroMQ message authentication."""

import unittest

from zmq_auth import sign_message, verify_message


class ZmqAuthTests(unittest.TestCase):
    """Validate signing and verification of rollout RPC frames."""

    def test_round_trip(self):
        """A signed message verifies and returns its payload."""
        key = b"k" * 32
        payload = b"serialized request"

        self.assertEqual(verify_message(key, 0, sign_message(key, 0, payload, "request"), "request"), payload)

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


if __name__ == "__main__":
    unittest.main()

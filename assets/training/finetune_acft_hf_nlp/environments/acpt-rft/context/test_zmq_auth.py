# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for vLLM rollout ZeroMQ message authentication."""

import ast
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import ipaddress
import os
from pathlib import Path
import socket
import ssl
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

from zmq_auth import (
    sign_message,
    validate_ray_tls_for_addresses,
    verify_message,
    verify_ray_worker_tls,
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

    def test_tcp_rejects_invalid_ray_tls_files(self):
        """File existence alone does not establish valid TLS credentials."""
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
                with self.assertRaises(RuntimeError):
                    validate_ray_tls_for_addresses(["tcp://10.0.0.2:1234"])


class RayTlsTests(unittest.TestCase):
    """Verify key release against real TLS and plaintext listeners."""

    @classmethod
    def setUpClass(cls):
        """Create a local test CA and certificate for mutual TLS."""
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        subject = x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, "rollout-test"),
        ])
        now = datetime.now(timezone.utc)
        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(subject)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - timedelta(minutes=1))
            .not_valid_after(now + timedelta(days=1))
            .add_extension(x509.BasicConstraints(ca=True, path_length=None),
                           critical=True)
            .add_extension(x509.SubjectAlternativeName([
                x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
            ]), critical=False)
            .sign(key, hashes.SHA256())
        )
        cert_path = Path(cls.directory.name) / "cert.pem"
        key_path = Path(cls.directory.name) / "key.pem"
        cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
        key_path.write_bytes(key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        ))
        cls.environment = {
            "RAY_USE_TLS": "1",
            "RAY_TLS_SERVER_CERT": str(cert_path),
            "RAY_TLS_SERVER_KEY": str(key_path),
            "RAY_TLS_CA_CERT": str(cert_path),
        }

    @contextmanager
    def listener(self, tls):
        """Serve a TLS or plaintext endpoint without needing GPUs or Ray."""
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(
            self.environment["RAY_TLS_SERVER_CERT"],
            self.environment["RAY_TLS_SERVER_KEY"],
        )
        context.load_verify_locations(
            self.environment["RAY_TLS_CA_CERT"],
        )
        context.verify_mode = ssl.CERT_REQUIRED
        errors = []
        with socket.socket() as server:
            server.bind(("127.0.0.1", 0))
            server.listen()
            server.settimeout(5)

            def serve():
                try:
                    with server.accept()[0] as connection:
                        connection.settimeout(5)
                        if tls:
                            with context.wrap_socket(
                                connection, server_side=True,
                            ) as secure:
                                secure.recv(33)
                                secure.sendall(b"\x00")
                        else:
                            connection.sendall(b"not a TLS listener")
                except Exception as exc:
                    errors.append(exc)

            thread = threading.Thread(target=serve, daemon=True)
            thread.start()
            try:
                yield server.getsockname()
            finally:
                thread.join(6)
                self.assertFalse(thread.is_alive())
                self.assertEqual(errors, [])

    def credentials(self):
        """Load the actual worker key-return method without GPU imports."""
        source = Path(__file__).with_name("vllm_rollout").read_text(
            encoding="utf-8",
        )
        rollout = next(
            node for node in ast.parse(source).body
            if isinstance(node, ast.ClassDef)
            and node.name == "vLLMAsyncRollout"
        )
        method = next(
            node for node in rollout.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "get_zeromq_address"
        )
        namespace = {"verify_ray_worker_tls": verify_ray_worker_tls}
        exec(compile(ast.Module(body=[method], type_ignores=[]),
                     "vllm_rollout", "exec"), namespace)
        worker = SimpleNamespace(
            address="tcp://127.0.0.1:1234",
            _zmq_auth_key=b"k" * 32,
        )
        return namespace["get_zeromq_address"](worker)

    def test_refuses_key_without_tls(self):
        """Key release fails even when the caller uses effective nnodes=1."""
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "RAY_USE_TLS"):
                self.credentials()

    def test_refuses_key_from_plaintext_ray(self):
        """Late TLS settings cannot disguise a plaintext listener."""
        with self.listener(tls=False) as address:
            with (
                patch.dict(os.environ, self.environment, clear=True),
                patch("zmq_auth._ray_worker_rpc_address",
                      return_value=address),
            ):
                with self.assertRaisesRegex(RuntimeError, "verified TLS"):
                    self.credentials()

    def test_releases_key_after_mutual_tls(self):
        """A worker releases its key only after a real verified handshake."""
        with self.listener(tls=True) as address:
            with (
                patch.dict(os.environ, self.environment, clear=True),
                patch("zmq_auth._ray_worker_rpc_address",
                      return_value=address),
            ):
                self.assertEqual(
                    self.credentials(),
                    ("tcp://127.0.0.1:1234", (b"k" * 32).hex()),
                )

    def test_ipc_does_not_require_ray_tls(self):
        """Local IPC preserves its existing behavior without TLS settings."""
        with patch.dict(os.environ, {}, clear=True):
            verify_ray_worker_tls("ipc:///tmp/rollout-test.ipc")


if __name__ == "__main__":
    unittest.main()

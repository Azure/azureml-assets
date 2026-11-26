# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains unit tests for vesta image modifier."""

import io
import os
from pathlib import Path
from urllib.parse import urlparse

import pytest
import requests
from mock import MagicMock

from src.batch_score.common.request_modification.modifiers.vesta_image_encoder import (
    FolderNotMounted,
    ImageEncoder,
    UnsafeUrl,
    UnsuccessfulUrlResponse,
    VestaImageModificationException,
)
from src.batch_score.common.request_modification.modifiers.vesta_image_modifier import (
    VestaImageModifier,
)
from tests.fixtures.vesta_image_modifier import (
    MOCKED_BINARY_FROM_FILE,
    MOCKED_BINARY_FROM_URL,
)


def _mock_http(monkeypatch, routes):
    """Serve canned (status_code, headers, content) responses by url and return the list of sent requests."""
    sent_requests = []

    def send(adapter, request, **kwargs):
        sent_requests.append(request)
        status_code, headers, content = routes[request.url]
        response = requests.Response()
        response.status_code = status_code
        response.headers.update(headers)
        response.raw = io.BytesIO(content)
        response.url = request.url
        response.request = request
        return response

    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", send)
    return sent_requests


@pytest.fixture(autouse=True)
def _no_environment_proxies(monkeypatch):
    """Ignore proxies configured on the machine running the tests."""
    monkeypatch.setattr(requests.utils, "getproxies", lambda: {})


def test_vesta_payload_type():
    """Test vesta payload tpye."""
    assert VestaImageModifier.vesta_payload_type(request_obj={"transcript": []}) == "transcript"
    assert VestaImageModifier.vesta_payload_type(request_obj={"prompt": []}) == "prompt"
    assert VestaImageModifier.vesta_payload_type(request_obj={"foobar123": []}) is None
    assert VestaImageModifier.vesta_payload_type(request_obj={"prompt": [], "transcript": []}) is None


def test_is_vesta_payload():
    """Test is vesta payload."""
    assert VestaImageModifier.is_vesta_payload(request_obj={"prompt": [{"type": "text", "data": "foobar"}]}) is True
    assert VestaImageModifier.is_vesta_payload(
        request_obj={"transcript": [{"type": "text", "data": "foobar"}]}) is True
    assert VestaImageModifier.is_vesta_payload(
        request_obj={"transcript": [{"type": "text", "data": "foobar"}, {"hello": "world"}]}) is False
    assert VestaImageModifier.is_vesta_payload(request_obj={"transcript": [{"hello": "world"}]}) is False
    assert VestaImageModifier.is_vesta_payload(request_obj={"prompt": [{"hello": "world"}]}) is False
    assert VestaImageModifier.is_vesta_payload(request_obj={"invalid": [{"hello": "world"}]}) is False


def test_modify(mock_get_logger, make_vesta_image_modifier, mock__b64_from_url):
    """Test modify."""
    vesta_request_obj = {
        "transcript": [
            {
                "type": "text",
                "data": "This is just a test. These aren't even images"
            },
            {
                "type": "image",
                "data": "ImageUrl!https://fake.url"
            },
            {
                "type": "image",
                "data": "ImageUrl!https://another.fake.url"
            },
            {
                "type": "image",
                "data": "/9j/4AAQSkZJRgABAQAAAQABAAD/"
            },
            {
                "type": "image_hr",
                "data": "/9j/4AAQSkZJRgABAQAAAQABAAD/"
            },
            {
                "type": "text",
                "data": "End of the test"
            }]}
    vesta_image_modifier: VestaImageModifier = make_vesta_image_modifier()

    modified_request_obj = vesta_image_modifier.modify(request_obj=vesta_request_obj)

    # Assert the two URLs were called
    assert any("fake.url" in urlparse(url).hostname for url in mock__b64_from_url)
    assert any("another.fake.url" in urlparse(url).hostname for url in mock__b64_from_url)

    # Assert modifications are correct
    assert modified_request_obj["transcript"][1]["data"] == MOCKED_BINARY_FROM_URL
    assert modified_request_obj["transcript"][2]["data"] == MOCKED_BINARY_FROM_URL
    assert modified_request_obj["transcript"][3]["data"] == "/9j/4AAQSkZJRgABAQAAAQABAAD/"
    assert modified_request_obj["transcript"][4]["data"] == "/9j/4AAQSkZJRgABAQAAAQABAAD/"


def test_modify_invalid_image(mock_get_logger, make_vesta_image_modifier, mock_encode_b64):
    """Test modify invalid image."""
    vesta_request_obj = {"transcript": [{"type": "image", "data": "invalid_data"}]}
    vesta_image_modifier: VestaImageModifier = make_vesta_image_modifier()

    mock_encode_b64["exception"] = FolderNotMounted()
    with pytest.raises(VestaImageModificationException):
        vesta_image_modifier.modify(vesta_request_obj)

    mock_encode_b64["exception"] = Exception()
    with pytest.raises(VestaImageModificationException):
        vesta_image_modifier.modify(vesta_request_obj)


def test_encode_b64_file(mock_get_logger, make_image_encoder, mock__b64_from_file):
    """Test encode b64 file."""
    path_prefix = "E:/fake/folder/"
    image_encoder: ImageEncoder = make_image_encoder(image_input_folder_str=path_prefix)

    image_target = "target.png"
    image_data = f"ImageFile!{image_target}"
    encoding = image_encoder.encode_b64(image_data=image_data)

    assert encoding == MOCKED_BINARY_FROM_FILE
    assert mock__b64_from_file[-1] == os.path.join(str(Path(path_prefix)), str(Path(image_target)))
    assert mock_get_logger.debug.called or mock_get_logger.info.called

    image_encoder: ImageEncoder = make_image_encoder(image_input_folder_str=None)
    with pytest.raises(FolderNotMounted):
        image_encoder.encode_b64(image_data=image_data)


def test_encode_b64_url(mock_get_logger, make_image_encoder, mock__b64_from_url):
    """Test encode b64 url."""
    image_encoder: ImageEncoder = make_image_encoder(image_input_folder_str=None)

    image_target = "https://fake.url"
    image_data = f"ImageUrl!{image_target}"
    encoding = image_encoder.encode_b64(image_data=image_data)

    assert encoding == MOCKED_BINARY_FROM_URL
    assert mock__b64_from_url[-1] == image_target
    assert mock_get_logger.debug.called or mock_get_logger.info.called


def test__b64_from_url(mock_get_logger: MagicMock, monkeypatch, make_image_encoder):
    """Test b64 from url."""
    image_encoder: ImageEncoder = make_image_encoder()
    target_url = "https://fake.url/"

    # Resolve to a public address so the SSRF validation permits the request.
    monkeypatch.setattr(
        "socket.getaddrinfo",
        lambda host, port, **kwargs: [(2, 1, 6, "", ("93.184.216.34", port or 443))])

    _mock_http(monkeypatch, {target_url: (200, {}, b"MOCK_CONTENT_BYTES")})

    encoding = image_encoder._b64_from_url(target_url)
    assert encoding

    _mock_http(monkeypatch, {target_url: (404, {}, b"")})

    with pytest.raises(UnsuccessfulUrlResponse):
        image_encoder._b64_from_url(target_url)
    assert mock_get_logger.info.called
    assert target_url in mock_get_logger.info.call_args.args[0]


def test__b64_from_url_rejects_disallowed_scheme(make_image_encoder):
    """Test that non-http(s) urls are rejected to mitigate SSRF (CWE-918)."""
    image_encoder: ImageEncoder = make_image_encoder()

    for url in ["file:///etc/passwd", "ftp://example.com/image.png", "gopher://example.com"]:
        with pytest.raises(UnsafeUrl):
            image_encoder._b64_from_url(url)


def test__b64_from_url_rejects_non_public_address(monkeypatch, make_image_encoder):
    """Test that urls resolving to non-public addresses are rejected to mitigate SSRF (CWE-918)."""
    image_encoder: ImageEncoder = make_image_encoder()
    sent_requests = _mock_http(monkeypatch, {})

    blocked_ips = [
        "169.254.169.254",  # cloud instance metadata endpoint (link-local)
        "168.63.129.16",    # Azure platform endpoint (WireServer)
        "127.0.0.1",        # loopback
        "10.0.0.5",         # private
        "192.168.1.1",      # private
        "100.64.0.1",       # shared address space
        "192.0.0.8",        # IETF protocol assignment (global on unpatched Python releases)
        "0.0.0.0",          # unspecified
        "::ffff:127.0.0.1",  # IPv4-mapped loopback
        "3fff::1",          # IPv6 documentation
    ]
    for ip in blocked_ips:
        monkeypatch.setattr(
            "socket.getaddrinfo",
            lambda host, port, _ip=ip, **kwargs: [(2, 1, 6, "", (_ip, port or 443))])
        with pytest.raises(UnsafeUrl):
            image_encoder._b64_from_url("https://malicious.example/image.png")
    assert sent_requests == []


def test__b64_from_url_rejects_proxy(monkeypatch, make_image_encoder):
    """Test that urls are not fetched through a proxy, where the destination cannot be pinned."""
    image_encoder: ImageEncoder = make_image_encoder()
    monkeypatch.setattr(requests.utils, "getproxies", lambda: {"https": "http://proxy.example:3128"})
    monkeypatch.setattr(
        "socket.getaddrinfo",
        lambda host, port, **kwargs: [(2, 1, 6, "", ("93.184.216.34", port or 443))])
    sent_requests = _mock_http(monkeypatch, {})

    with pytest.raises(UnsafeUrl):
        image_encoder._b64_from_url("https://public.example/image.png")
    assert sent_requests == []


def test__b64_from_url_rejects_redirect_to_non_public_address(monkeypatch, make_image_encoder):
    """Test that every redirect hop is validated to mitigate SSRF (CWE-918)."""
    image_encoder: ImageEncoder = make_image_encoder()
    image_url = "https://public.example/image.png"
    internal_url = "http://internal.example/secret.png"
    monkeypatch.setattr(
        "socket.getaddrinfo",
        lambda host, port, **kwargs: [
            (2, 1, 6, "", ("10.0.0.5" if host == "internal.example" else "93.184.216.34", port or 443))])
    sent_requests = _mock_http(monkeypatch, {
        image_url: (302, {"Location": internal_url}, b""),
        internal_url: (200, {}, b"internal"),
    })

    with pytest.raises(UnsafeUrl):
        image_encoder._b64_from_url(image_url)
    assert [request.url for request in sent_requests] == [image_url]


def test__b64_from_url_keeps_credentials_only_on_same_origin_redirect(monkeypatch, make_image_encoder):
    """Test that url credentials are kept on same-origin redirects and removed when the host changes."""
    image_encoder: ImageEncoder = make_image_encoder()
    image_url = "https://user:pass@public.example/image.png"
    same_origin_url = "https://public.example/v2/image.png"
    other_origin_url = "https://cdn.example/image.png"
    monkeypatch.setattr(
        "socket.getaddrinfo",
        lambda host, port, **kwargs: [(2, 1, 6, "", ("93.184.216.34", port or 443))])
    sent_requests = _mock_http(monkeypatch, {
        image_url: (302, {"Location": same_origin_url}, b""),
        same_origin_url: (302, {"Location": other_origin_url}, b""),
        other_origin_url: (200, {}, b"image"),
    })

    assert image_encoder._b64_from_url(image_url)
    assert ["Authorization" in request.headers for request in sent_requests] == [True, True, False]


def test__b64_from_url_removes_credentials_on_backslash_redirect(monkeypatch, make_image_encoder):
    """Test that url credentials are removed when a redirect only appears to stay on the same host."""
    image_encoder: ImageEncoder = make_image_encoder()
    image_url = "https://user:pass@public.example/image.png"
    location = "https://attacker.example\\@public.example/image.png"
    attacker_url = requests.Request("GET", location).prepare().url
    monkeypatch.setattr(
        "socket.getaddrinfo",
        lambda host, port, **kwargs: [(2, 1, 6, "", ("93.184.216.34", port or 443))])
    sent_requests = _mock_http(monkeypatch, {
        image_url: (302, {"Location": location}, b""),
        attacker_url: (200, {}, b"image"),
    })

    assert image_encoder._b64_from_url(image_url)
    assert [request.url for request in sent_requests] == [image_url, attacker_url]
    assert "Authorization" not in sent_requests[1].headers


def test__b64_from_url_rejects_proxy_for_the_host_requests_connects_to(monkeypatch, make_image_encoder):
    """Test that the proxy check uses the host requests connects to, not the one urlparse reports."""
    image_encoder: ImageEncoder = make_image_encoder()
    monkeypatch.delenv("no_proxy", raising=False)
    monkeypatch.setenv("NO_PROXY", "public.example")
    monkeypatch.setattr(requests.utils, "getproxies", lambda: {"https": "http://proxy.example:3128"})
    monkeypatch.setattr(
        "socket.getaddrinfo",
        lambda host, port, **kwargs: [(2, 1, 6, "", ("93.184.216.34", port or 443))])
    sent_requests = _mock_http(monkeypatch, {})

    with pytest.raises(UnsafeUrl):
        image_encoder._b64_from_url("https://attacker.example\\@public.example/image.png")
    assert sent_requests == []


def test__b64_from_url_rejects_dns_rebinding(monkeypatch, make_image_encoder):
    """Test that connections are pinned to validated addresses to mitigate DNS rebinding."""
    image_encoder: ImageEncoder = make_image_encoder()
    addresses = iter(["93.184.216.34"])
    monkeypatch.setattr(
        "socket.getaddrinfo",
        lambda host, port, **kwargs: [(2, 1, 6, "", (next(addresses, "127.0.0.1"), port or 443))])

    with pytest.raises(UnsafeUrl):
        image_encoder._b64_from_url("https://rebind.example/image.png")


def test__b64_from_url_rejects_unresolvable_host(monkeypatch, make_image_encoder):
    """Test that urls whose host cannot be resolved are rejected to mitigate SSRF (CWE-918)."""
    import socket

    image_encoder: ImageEncoder = make_image_encoder()

    def raise_gaierror(*args, **kwargs):
        raise socket.gaierror("name resolution failed")

    monkeypatch.setattr("socket.getaddrinfo", raise_gaierror)

    with pytest.raises(UnsafeUrl):
        image_encoder._b64_from_url("https://does-not-resolve.example/image.png")


def test_encode_b64_raw(mock_get_logger, make_image_encoder):
    """Test encode b64 raw."""
    image_encoder: ImageEncoder = make_image_encoder(image_input_folder_str=None)
    image_data = "MOCK_DATA"
    encoding = image_encoder.encode_b64(image_data=image_data)

    assert encoding == image_data
    assert mock_get_logger.debug.called or mock_get_logger.info.called

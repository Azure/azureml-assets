# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Test vision utility code."""

import base64
import io
import os
import pytest
import requests
import socket
import tempfile
import urllib3

import pandas as pd

from PIL import Image, ImageChops
from urllib.parse import urlparse

from azureml.model.mgmt.processors.common import vision_utils
from azureml.model.mgmt.processors.common.vision_utils import (
    process_image, process_image_pandas_series, save_image, _is_valid_url, IMAGE_URL_ALLOWED_NETWORKS_ENV_VAR,
    UnsafeImageUrlError
)

PUBLIC_ADDRESS = "20.70.246.20"


def _mock_dns(monkeypatch, resolve):
    """Make socket.getaddrinfo resolve every host name to the address or addresses returned by resolve(host)."""
    def getaddrinfo(host, port, *args, **kwargs):
        addresses = resolve(host)
        if isinstance(addresses, str):
            addresses = [addresses]
        return [
            (socket.AF_INET6 if ":" in address else socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "",
             (address, port))
            for address in addresses
        ]

    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)


def _use_netrc(monkeypatch, tmp_path):
    """Use a .netrc file with credentials for images.contoso.com and return their Authorization header."""
    netrc_path = tmp_path / "netrc"
    netrc_path.write_text("machine images.contoso.com login user password secret\n")
    monkeypatch.setenv("NETRC", str(netrc_path))
    return "Basic " + base64.b64encode(b"user:secret").decode()


def _mock_http(monkeypatch, routes):
    """Serve canned (status_code, headers, body) responses by url and return the list of sent requests."""
    sent_requests = []

    def send(adapter, request, **kwargs):
        sent_requests.append(request)
        status_code, headers, body = routes[request.url]
        response = requests.Response()
        response.status_code = status_code
        response.headers.update(headers)
        response.raw = body if hasattr(body, "read") else io.BytesIO(body)
        response.url = request.url
        response.request = request
        return response

    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", send)
    return sent_requests


class _UnreadableBody(io.BytesIO):
    """Response body that fails the test when it is read."""

    def read(self, *args, **kwargs):
        """Fail when the body is read."""
        raise AssertionError("The response body must not be read.")


@pytest.fixture(autouse=True)
def _no_environment_proxies(monkeypatch):
    """Ignore proxies configured on the machine running the tests."""
    monkeypatch.setattr(requests.utils, "getproxies", lambda: {})


class TestVisionUtilities:
    """Test vision utility functions."""

    def test_save_image(self):
        """Test that save_image() behaves correctly for png files."""
        image = Image.new("RGB", (640, 480))
        image.putpixel((320, 240), (0, 0, 255))
        image.putpixel((320, 241), (255, 255, 0))

        with tempfile.TemporaryDirectory() as temporary_directory_name:
            image_file_name = save_image(temporary_directory_name, image, "png")

            image2 = Image.open(os.path.join(temporary_directory_name, image_file_name))
            assert not ImageChops.difference(image, image2).getbbox()

    def test_process_image_bytes(self):
        """Test process_image() with a bytes object."""
        image_bytes = bytes([0, 1])

        assert process_image(image_bytes) == image_bytes

    def test_process_image_url_successful(self, monkeypatch):
        """Test process_image() with a URL whose host resolves to a public address."""
        image_url = "http://microsoft.com/image.jpg"
        image_bytes = bytes([2, 3])
        _mock_dns(monkeypatch, lambda host: PUBLIC_ADDRESS)
        sent_requests = _mock_http(monkeypatch, {image_url: (200, {}, image_bytes)})

        assert process_image(image_url) == image_bytes
        assert [request.url for request in sent_requests] == [image_url]

    def test_process_image_url_unavailable(self, monkeypatch):
        """Test process_image() with an 'incorrect' URL."""
        image_url = "http://microsoft.com/this_image_does_not_exist.jpg"
        _mock_dns(monkeypatch, lambda host: PUBLIC_ADDRESS)
        _mock_http(monkeypatch, {image_url: (404, {}, b"")})

        with pytest.raises(ValueError) as e:
            assert process_image(image_url)
        assert "Unable to retrieve image from url string due to exception: " in str(e.value)

    @pytest.mark.parametrize("address", [
        "127.0.0.1", "10.0.0.5", "172.16.0.1", "192.168.1.1", "100.64.0.1", "0.0.0.0", "169.254.169.254",
        "168.63.129.16", "192.0.0.8", "::1", "fd00::1", "fe80::1", "::ffff:10.0.0.5", "64:ff9b::a9fe:a9fe", "3fff::1",
    ])
    def test_process_image_url_rejects_non_public_address(self, monkeypatch, address):
        """Test that process_image() does not fetch URLs whose host resolves to a non-public address."""
        _mock_dns(monkeypatch, lambda host: address)
        sent_requests = _mock_http(monkeypatch, {})

        with pytest.raises(UnsafeImageUrlError):
            process_image("http://images.contoso.com/image.jpg")
        assert sent_requests == []

    def test_process_image_url_rejects_ip_literal(self, monkeypatch):
        """Test that process_image() does not fetch URLs with a non-public IP literal host."""
        sent_requests = _mock_http(monkeypatch, {})

        with pytest.raises(UnsafeImageUrlError):
            process_image("http://169.254.169.254/metadata/instance?api-version=2021-02-01&format=.json")
        assert sent_requests == []

    def test_process_image_url_rejects_proxy(self, monkeypatch):
        """Test that process_image() does not download images through a proxy."""
        monkeypatch.setattr(requests.utils, "getproxies", lambda: {"http": "http://proxy.contoso.com:3128"})
        _mock_dns(monkeypatch, lambda host: PUBLIC_ADDRESS)
        sent_requests = _mock_http(monkeypatch, {})

        with pytest.raises(UnsafeImageUrlError):
            process_image("http://images.contoso.com/image.jpg")
        assert sent_requests == []

    def test_process_image_url_rejects_proxy_for_the_host_requests_connects_to(self, monkeypatch):
        """Test that the proxy check uses the host requests connects to, not the one urlparse reports."""
        monkeypatch.delenv("no_proxy", raising=False)
        monkeypatch.setenv("NO_PROXY", "images.contoso.com")
        monkeypatch.setattr(requests.utils, "getproxies", lambda: {"http": "http://proxy.contoso.com:3128"})
        _mock_dns(monkeypatch, lambda host: PUBLIC_ADDRESS)
        sent_requests = _mock_http(monkeypatch, {})

        with pytest.raises(UnsafeImageUrlError):
            process_image("http://attacker.contoso.com\\@images.contoso.com/image.jpg")
        assert sent_requests == []

    def test_process_image_url_rejects_redirect_to_non_public_address(self, monkeypatch):
        """Test that process_image() validates the destination of every redirect."""
        image_url = "http://images.contoso.com/image.jpg"
        internal_url = "http://internal.contoso.com/secret.png"
        _mock_dns(monkeypatch, lambda host: "10.0.0.5" if host == "internal.contoso.com" else PUBLIC_ADDRESS)
        sent_requests = _mock_http(monkeypatch, {
            image_url: (302, {"Location": internal_url}, b""),
            internal_url: (200, {}, b"internal"),
        })

        with pytest.raises(UnsafeImageUrlError):
            process_image(image_url)
        assert [request.url for request in sent_requests] == [image_url]

    def test_process_image_url_follows_redirect_to_public_address(self, monkeypatch):
        """Test that process_image() follows relative redirects without reading the redirect response body."""
        image_url = "http://images.contoso.com/image.jpg"
        _mock_dns(monkeypatch, lambda host: PUBLIC_ADDRESS)
        sent_requests = _mock_http(monkeypatch, {
            image_url: (301, {"Location": "/v2/image.jpg"}, _UnreadableBody()),
            "http://images.contoso.com/v2/image.jpg": (200, {}, b"image"),
        })

        assert process_image(image_url) == b"image"
        assert [request.url for request in sent_requests] == [image_url, "http://images.contoso.com/v2/image.jpg"]

    def test_process_image_url_keeps_credentials_only_on_same_origin_redirect(self, monkeypatch):
        """Test that URL credentials are kept on same-origin redirects and removed when the host changes."""
        image_url = "http://user:pass@images.contoso.com/image.jpg"
        same_origin_url = "http://images.contoso.com/v2/image.jpg"
        other_origin_url = "http://cdn.contoso.com/image.jpg"
        _mock_dns(monkeypatch, lambda host: PUBLIC_ADDRESS)
        sent_requests = _mock_http(monkeypatch, {
            image_url: (302, {"Location": same_origin_url}, b""),
            same_origin_url: (302, {"Location": other_origin_url}, b""),
            other_origin_url: (200, {}, b"image"),
        })

        assert process_image(image_url) == b"image"
        assert ["Authorization" in request.headers for request in sent_requests] == [True, True, False]

    def test_process_image_url_removes_credentials_on_backslash_redirect(self, monkeypatch):
        """Test that credentials are removed when a redirect only appears to stay on the same host."""
        image_url = "http://user:pass@images.contoso.com/image.jpg"
        location = "http://attacker.contoso.com\\@images.contoso.com/image.jpg"
        attacker_url = requests.Request("GET", location).prepare().url
        _mock_dns(monkeypatch, lambda host: PUBLIC_ADDRESS)
        sent_requests = _mock_http(monkeypatch, {
            image_url: (302, {"Location": location}, b""),
            attacker_url: (200, {}, b"image"),
        })

        assert process_image(image_url) == b"image"
        assert [request.url for request in sent_requests] == [image_url, attacker_url]
        assert "Authorization" not in sent_requests[1].headers

    def test_process_image_url_rejects_redirect_loop(self, monkeypatch):
        """Test that process_image() limits the number of redirects."""
        image_url = "http://images.contoso.com/image.jpg"
        _mock_dns(monkeypatch, lambda host: PUBLIC_ADDRESS)
        _mock_http(monkeypatch, {image_url: (302, {"Location": image_url}, b"")})

        with pytest.raises(ValueError, match="maximum of 5 redirects"):
            process_image(image_url)

    def test_process_image_url_rejects_dns_rebinding(self, monkeypatch):
        """Test that connections are pinned to validated addresses when DNS changes after validation."""
        addresses = iter([PUBLIC_ADDRESS])
        _mock_dns(monkeypatch, lambda host: next(addresses, "127.0.0.1"))

        with pytest.raises(UnsafeImageUrlError):
            process_image("http://rebind.contoso.com/image.jpg")

    @pytest.mark.parametrize("addresses", [[PUBLIC_ADDRESS, "10.0.0.5"], ["10.0.0.5", PUBLIC_ADDRESS]])
    def test_process_image_url_rejects_host_with_any_non_public_address(self, monkeypatch, addresses):
        """Test that process_image() rejects a host that resolves to both public and non-public addresses."""
        _mock_dns(monkeypatch, lambda host: addresses)
        sent_requests = _mock_http(monkeypatch, {})

        with pytest.raises(UnsafeImageUrlError):
            process_image("http://images.contoso.com/image.jpg")
        assert sent_requests == []

    @pytest.mark.parametrize("connection_class, port", [
        (vision_utils._AllowedAddressHTTPConnection, 80),
        (vision_utils._AllowedAddressHTTPSConnection, 443),
    ])
    def test_connection_is_pinned_to_validated_addresses(self, monkeypatch, connection_class, port):
        """Test that connections only go to validated addresses, fall back between them and keep the host name."""
        answers = iter([[PUBLIC_ADDRESS, "20.70.246.21"]])
        _mock_dns(monkeypatch, lambda host: next(answers, "127.0.0.1"))
        destinations = []

        def create_connection(address, *args, **kwargs):
            destinations.append(address)
            if address[0] == PUBLIC_ADDRESS:
                raise OSError("unreachable")
            return "socket"

        monkeypatch.setattr(urllib3.util.connection, "create_connection", create_connection)
        connection = connection_class("images.contoso.com", port)

        assert connection._new_conn() == "socket"
        assert destinations == [(PUBLIC_ADDRESS, port), ("20.70.246.21", port)]
        assert connection.host == "images.contoso.com"

    @pytest.mark.parametrize("image_url, location", [
        ("http://attacker.contoso.com\\@images.contoso.com/image.jpg", None),
        ("http://images.contoso.com:ignored@attacker.contoso.com/image.jpg", None),
        ("http://images.contoso.com/image.jpg", "http://attacker.contoso.com\\@images.contoso.com/image.jpg"),
        ("http://images.contoso.com/image.jpg", "http://images.contoso.com:ignored@attacker.contoso.com/image.jpg"),
    ])
    def test_process_image_url_sends_netrc_credentials_only_to_their_host(
            self, monkeypatch, tmp_path, image_url, location):
        """Test that .netrc credentials are not sent to a host that a url only appears to name."""
        netrc_authorization = _use_netrc(monkeypatch, tmp_path)
        _mock_dns(monkeypatch, lambda host: PUBLIC_ADDRESS)
        first_url = requests.Request("GET", image_url).prepare().url
        routes = {first_url: (302, {"Location": location}, b"") if location else (200, {}, b"image")}
        if location:
            routes[requests.Request("GET", location).prepare().url] = (200, {}, b"image")
        sent_requests = _mock_http(monkeypatch, routes)

        assert process_image(image_url) == b"image"
        attacker_requests = [r for r in sent_requests if urlparse(r.url).hostname == "attacker.contoso.com"]
        assert len(attacker_requests) == 1
        assert attacker_requests[0].headers.get("Authorization") != netrc_authorization

    def test_process_image_url_sends_netrc_credentials_to_their_host(self, monkeypatch, tmp_path):
        """Test that .netrc credentials are still sent to the host they belong to."""
        netrc_authorization = _use_netrc(monkeypatch, tmp_path)
        image_url = "http://images.contoso.com/image.jpg"
        _mock_dns(monkeypatch, lambda host: PUBLIC_ADDRESS)
        sent_requests = _mock_http(monkeypatch, {image_url: (200, {}, b"image")})

        assert process_image(image_url) == b"image"
        assert sent_requests[0].headers.get("Authorization") == netrc_authorization

    def test_process_image_url_allowed_networks(self, monkeypatch):
        """Test that deployments can allow private networks, but never link-local addresses."""
        monkeypatch.setenv(IMAGE_URL_ALLOWED_NETWORKS_ENV_VAR, "10.1.0.0/16, 169.254.0.0/16")
        image_url = "http://blob.contoso.com/image.jpg"
        _mock_dns(monkeypatch, lambda host: "169.254.169.254" if host == "metadata.contoso.com" else "10.1.2.3")
        _mock_http(monkeypatch, {image_url: (200, {}, b"image")})

        assert process_image(image_url) == b"image"
        with pytest.raises(UnsafeImageUrlError):
            process_image("http://metadata.contoso.com/image.jpg")

    @pytest.mark.parametrize("allowed_network, address", [
        ("64:ff9b:1::/48", "64:ff9b:1::a9fe:a9fe"),
        ("2001::/32", "2001:0:4136:e378:8000:63bf:3fff:fdd2"),
        ("64:ff9b::/96", "64:ff9b::a9fe:a9fe"),
        ("2002::/16", "2002:a9fe:a9fe::1"),
    ])
    def test_process_image_url_allowed_networks_cannot_admit_blocked_destinations(
            self, monkeypatch, allowed_network, address):
        """Test that allowing IPv6 translation ranges cannot admit an always-blocked IPv4 destination."""
        monkeypatch.setenv(IMAGE_URL_ALLOWED_NETWORKS_ENV_VAR, allowed_network)
        _mock_dns(monkeypatch, lambda host: address)
        sent_requests = _mock_http(monkeypatch, {})

        with pytest.raises(UnsafeImageUrlError):
            process_image("http://images.contoso.com/image.jpg")
        assert sent_requests == []

    def test_process_image_url_allowed_networks_match_the_resolved_address(self, monkeypatch):
        """Test that allowed networks are matched against the resolved address, not an embedded IPv4 address."""
        image_url = "http://images.contoso.com/image.jpg"
        _mock_http(monkeypatch, {image_url: (200, {}, b"image")})

        monkeypatch.setenv(IMAGE_URL_ALLOWED_NETWORKS_ENV_VAR, "10.1.0.0/16")
        _mock_dns(monkeypatch, lambda host: "2002:a01:203::1")
        with pytest.raises(UnsafeImageUrlError):
            process_image(image_url)

        monkeypatch.setenv(IMAGE_URL_ALLOWED_NETWORKS_ENV_VAR, "64:ff9b::/96")
        _mock_dns(monkeypatch, lambda host: "64:ff9b::a01:203")
        assert process_image(image_url) == b"image"

    def test_process_image_url_rejects_oversized_image(self, monkeypatch):
        """Test that process_image() limits the size of downloaded images."""
        image_url = "http://images.contoso.com/image.jpg"
        monkeypatch.setattr(vision_utils, "_IMAGE_URL_MAX_BYTES", 4)
        _mock_dns(monkeypatch, lambda host: PUBLIC_ADDRESS)
        _mock_http(monkeypatch, {image_url: (200, {}, b"12345")})

        with pytest.raises(ValueError, match="exceeds the maximum size"):
            process_image(image_url)

    def test_process_image_b64(self):
        """Test process_image() with a b64encoded string."""
        image_bytes = b"45"
        image_str = base64.encodebytes(image_bytes).decode("utf-8")

        assert process_image(image_str) == image_bytes

    def test_process_image_url_invalid(self):
        """Test process_image() with an invalid URL."""
        image_url = "ftp://microsoft.com/image.jpg"

        with pytest.raises(ValueError) as e:
            assert process_image(image_url)
        assert "The provided image string cannot be decoded. Expected format is base64 string or url string." in str(
            e.value
        )

    def test_process_image_list(self):
        """Test process_image() with an invalid image type."""
        image_array = [6, 7]

        with pytest.raises(ValueError) as e:
            assert process_image(image_array)
        assert (
            "Image received in <class 'list'> format which is not supported. "
            "Expected format is bytes, base64 string or url string."
        ) in str(e.value)

    def test_process_image_pandas_series1(self):
        """Test process_image_pandas_series() with a b64encoded string."""
        image_bytes = b"89"
        image_str = base64.encodebytes(image_bytes).decode("utf-8")

        assert (process_image_pandas_series(pd.Series(image_str)) == pd.Series(image_bytes)).all()

    def test_process_image_pandas_series2(self):
        """Test process_image_pandas_series() with similar usage as in the CLIP wrapper."""
        input_data = pd.DataFrame(
            columns=["image", "text"],
            data=[
                [base64.encodebytes(b"image with things").decode("utf-8"), "thing1,thing2"],
                [base64.encodebytes(b"image with stuff").decode("utf-8"), "stuff"],
                [base64.encodebytes(b"blank image").decode("utf-8"), "nothing"],
            ]
        )

        expected_decoded_images = [b"image with things", b"image with stuff", b"blank image"]

        decoded_images = input_data.loc[:, ["image"]].apply(axis=1, func=process_image_pandas_series)
        for decoded_image, expected_decoded_image in zip(decoded_images.iloc[:, 0], expected_decoded_images):
            assert decoded_image == expected_decoded_image

    def test_is_valid_url_llava_example(self):
        """Test that the function validating the URL validates the LLaVA example URL."""
        assert _is_valid_url("https://llava-vl.github.io/static/images/view.jpg")

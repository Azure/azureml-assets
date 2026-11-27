# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Vesta image encoder."""

import base64
import ipaddress
import os
import socket
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.connection import HTTPConnection, HTTPSConnection
from urllib3.connectionpool import HTTPConnectionPool, HTTPSConnectionPool
from urllib3.exceptions import ConnectTimeoutError, NewConnectionError

from ...telemetry import logging_utils as lu
from .request_modifier import RequestModificationException

# Azure platform endpoints (WireServer / host agent) are globally routable but must never be fetched.
_BLOCKED_NETWORKS = (ipaddress.ip_network("168.63.129.16/32"),)
# Special-purpose ranges that are not globally reachable but that ipaddress reports as global on some Python
# patch releases (192.0.0.0/24 before 3.8.20, 3.10.15 and 3.12.4; 3fff::/20 on current releases).
_NON_PUBLIC_NETWORKS = (ipaddress.ip_network("192.0.0.0/24"), ipaddress.ip_network("3fff::/20"))
_NAT64_NETWORK = ipaddress.ip_network("64:ff9b::/96")


class ImageEncoder():
    """Vesta image encoder."""

    IMAGE_URL = "ImageUrl!"
    IMAGE_FILE = "ImageFile!"

    # SSRF (CWE-918) mitigation: only fetch http(s) urls that resolve to public addresses.
    ALLOWED_URL_SCHEMES = ("http", "https")
    URL_REQUEST_TIMEOUT_SECONDS = 30
    URL_MAX_REDIRECTS = requests.models.DEFAULT_REDIRECT_LIMIT  # 30, the limit requests applies

    def __init__(self, image_input_folder_str: str = None) -> None:
        """Initialize ImageEncoder."""
        self.__image_input_folder_str: str = None
        if image_input_folder_str:
            self.__image_input_folder_str = str(Path(image_input_folder_str))

    def encode_b64(self, image_data: str) -> str:
        """Encode the image data in base64 format."""
        # Image URL to fetch
        if image_data.startswith(ImageEncoder.IMAGE_URL):
            url = image_data[len(ImageEncoder.IMAGE_URL):]
            lu.get_logger().debug(f"Encoding from URL: {url}.")
            return self._b64_from_url(url)

        # Image File mounted
        elif image_data.startswith(ImageEncoder.IMAGE_FILE):
            if not self.__image_input_folder_str:
                raise FolderNotMounted()

            target_file_path_suffix = image_data[len(ImageEncoder.IMAGE_FILE):]
            target_file_path = str(Path(target_file_path_suffix))
            lu.get_logger().debug(f"Encoding from File: {target_file_path}.")
            file_path = os.path.join(self.__image_input_folder_str, target_file_path)

            return self._b64_from_file(file_path)

        # Inlined image data
        else:
            lu.get_logger().debug("Image is already encoded, no encoding necessary.")
            return image_data

    def _validate_url(self, url: str) -> None:
        """Validate a caller-supplied image url to mitigate SSRF (CWE-918).

        The image url originates from untrusted per-row input data. Only http(s)
        urls whose host resolves exclusively to public addresses are permitted.
        Requests targeting loopback, private, link-local (e.g. the cloud instance
        metadata endpoint at 169.254.169.254), reserved, multicast, unspecified,
        or Azure platform (168.63.129.16) addresses are rejected before any network
        call is made. Every redirect hop is validated the same way, and connections
        are pinned to the validated addresses to prevent DNS rebinding. Urls that
        would be fetched through a proxy are rejected, since the proxy resolves the
        host itself and the destination could not be pinned.
        """
        parsed = urlparse(url)

        if parsed.scheme not in ImageEncoder.ALLOWED_URL_SCHEMES:
            raise UnsafeUrl(url)

        hostname = parsed.hostname
        if not hostname:
            raise UnsafeUrl(url)

        if requests.utils.select_proxy(url, requests.utils.get_environ_proxies(url)):
            raise UnsafeUrl(url)

        _resolve_public_addresses(hostname, parsed.port)

    def _get_url(self, url: str) -> requests.Response:
        """Get a url, validating the destination of every redirect hop."""
        headers = {}
        with requests.Session() as session:
            adapter = _PublicAddressHTTPAdapter()
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            for hop in range(ImageEncoder.URL_MAX_REDIRECTS + 1):
                prepared_url = _prepare_url(url)
                self._validate_url(prepared_url)
                resp = session.get(
                    url, headers=headers, auth=_PreparedUrlAuth(use_url_credentials=hop == 0), allow_redirects=False,
                    timeout=ImageEncoder.URL_REQUEST_TIMEOUT_SECONDS)
                if not resp.is_redirect:
                    return resp
                redirect_url = urljoin(prepared_url, session.get_redirect_target(resp))
                # Keep credentials only where requests' own redirect handling would keep them.
                authorization = resp.request.headers.get("Authorization")
                keep_authorization = authorization and not session.should_strip_auth(
                    prepared_url, _prepare_url(redirect_url))
                headers = {"Authorization": authorization} if keep_authorization else {}
                url = redirect_url

        lu.get_logger().info(f"URL exceeded the maximum of {ImageEncoder.URL_MAX_REDIRECTS} redirects.")
        raise UnsuccessfulUrlResponse()

    def _b64_from_url(self, url: str) -> str:
        start = time.time()
        resp = self._get_url(url)
        if resp.status_code != 200:
            lu.get_logger().info(
                f"URL '{url}' responded with an unsuccessful response: {resp.status_code}, {resp.reason}.")
            raise UnsuccessfulUrlResponse()

        img = resp.content
        end = time.time()
        lu.get_logger().debug(f"URL request latency: {end - start}")

        encoded_string = base64.b64encode(img).decode()
        return encoded_string

    def _b64_from_file(self, path: str) -> str:
        encoded_string: str = None

        start = time.time()
        with open(path, "rb") as image_file:
            image = image_file.read()
            end = time.time()
            lu.get_logger().debug(f"File request latency: {end - start}")

            encoded_string = base64.b64encode(image).decode()

        return encoded_string


class UnsuccessfulUrlResponse(Exception):
    """Unsuccessful url response."""

    def __init__(self, *args: object) -> None:
        """Initialize UnsuccessfulUrlResponse."""
        super().__init__(f"{ImageEncoder.IMAGE_URL} used in data, but url did not respond succesfully.")


class UnsafeUrl(Exception):
    """Unsafe url rejected by SSRF protection."""

    def __init__(self, url: str, *args: object) -> None:
        """Initialize UnsafeUrl."""
        super().__init__(
            f"{ImageEncoder.IMAGE_URL} used in data, but the url is not permitted. "
            "Only http(s) urls that resolve to public addresses are allowed.")


class FolderNotMounted(Exception):
    """Folder not mounted."""

    def __init__(self, *args: object) -> None:
        """Initialize FolderNotMounted."""
        super().__init__(f"{ImageEncoder.IMAGE_FILE} used in data, but no folder is mounted.")


class VestaImageModificationException(RequestModificationException):
    """Vesta image modification exception."""

    def __init__(self) -> None:
        """Initialize VestaImageModificationException."""
        super().__init__("The ImageEncoder raised an exception")


def _is_public_address(address: str) -> bool:
    """Check whether an ip address is a public destination that images may be fetched from."""
    ip = ipaddress.ip_address(address.split("%", 1)[0])
    if ip.version == 6:
        # Validate the IPv4 address embedded in IPv4-mapped, 6to4 and NAT64 addresses.
        if ip.ipv4_mapped is not None:
            ip = ip.ipv4_mapped
        elif ip.sixtofour is not None:
            ip = ip.sixtofour
        elif ip in _NAT64_NETWORK:
            ip = ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)
    if (
        ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_unspecified
        or any(ip in network for network in _BLOCKED_NETWORKS)
    ):
        return False
    return (
        ip.is_global
        and not ip.is_reserved
        and not (ip.version == 6 and ip.is_site_local)
        and not any(ip in network for network in _NON_PUBLIC_NETWORKS)
    )


def _prepare_url(url: str) -> str:
    """Return a url as requests sends it.

    requests and urlparse can disagree on the host of an unusual url (for example,
    one with a backslash in the authority), so destination, proxy and origin checks
    use the prepared url.
    """
    request = requests.PreparedRequest()
    try:
        request.prepare_url(url, None)
    except requests.exceptions.RequestException as e:
        raise UnsafeUrl(url) from e
    return request.url


class _PreparedUrlAuth(requests.auth.AuthBase):
    """Apply .netrc or url credentials for the host that requests connects to.

    requests looks up .netrc credentials for the url before preparing it, which
    can name a different host than the one it connects to. Passing this explicit
    auth disables that lookup. As in requests' redirect handling, .netrc
    credentials replace a carried Authorization header, and url credentials are
    only used for the original url.
    """

    def __init__(self, use_url_credentials: bool):
        """Initialize _PreparedUrlAuth."""
        self._use_url_credentials = use_url_credentials

    def __call__(self, request: requests.PreparedRequest) -> requests.PreparedRequest:
        """Add Basic credentials for the host of the prepared url, if there are any."""
        parsed = urlparse(request.url)
        credentials = requests.utils.get_netrc_auth(parsed._replace(netloc=parsed.netloc.rpartition("@")[2]).geturl())
        if not credentials and self._use_url_credentials:
            credentials = requests.utils.get_auth_from_url(request.url)
        if credentials and any(credentials):
            return requests.auth.HTTPBasicAuth(*credentials)(request)
        return request


def _resolve_public_addresses(hostname: str, port) -> list:
    """Resolve a host, ensuring that every address it resolves to is public."""
    try:
        address_infos = socket.getaddrinfo(hostname, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as e:
        raise UnsafeUrl(hostname) from e

    addresses = []
    for address_info in address_infos:
        address = address_info[4][0]
        if not _is_public_address(address):
            raise UnsafeUrl(hostname)
        if address not in addresses:
            addresses.append(address)

    if not addresses:
        raise UnsafeUrl(hostname)
    return addresses


def _connect_to_public_address(connection: HTTPConnection, connect):
    """Open a urllib3 connection pinned to a public address of its host.

    The host is resolved and validated again at connect time, so DNS rebinding
    cannot redirect the connection to another address after validation.
    """
    host = connection._dns_host
    addresses = _resolve_public_addresses(host.strip("[]"), connection.port)
    error = None
    for address in addresses:
        connection._dns_host = address
        try:
            return connect()
        except (NewConnectionError, ConnectTimeoutError) as e:
            error = e
        finally:
            connection._dns_host = host
    raise error


class _PublicAddressHTTPConnection(HTTPConnection):
    """urllib3 HTTP connection that only connects to public addresses."""

    def _new_conn(self):
        return _connect_to_public_address(self, super()._new_conn)


class _PublicAddressHTTPSConnection(HTTPSConnection):
    """urllib3 HTTPS connection that only connects to public addresses."""

    def _new_conn(self):
        return _connect_to_public_address(self, super()._new_conn)


class _PublicAddressHTTPConnectionPool(HTTPConnectionPool):
    """urllib3 HTTP connection pool that only connects to public addresses."""

    ConnectionCls = _PublicAddressHTTPConnection


class _PublicAddressHTTPSConnectionPool(HTTPSConnectionPool):
    """urllib3 HTTPS connection pool that only connects to public addresses."""

    ConnectionCls = _PublicAddressHTTPSConnection


class _PublicAddressHTTPAdapter(HTTPAdapter):
    """Requests adapter whose connections are pinned to public addresses.

    Urls that would be fetched through a proxy are rejected by
    ImageEncoder._validate_url, so every request uses a pinned connection.
    """

    def init_poolmanager(self, *args, **kwargs):
        """Initialize the pool manager with connection pools that validate addresses at connect time."""
        super().init_poolmanager(*args, **kwargs)
        self.poolmanager.pool_classes_by_scheme = {
            "http": _PublicAddressHTTPConnectionPool,
            "https": _PublicAddressHTTPSConnectionPool,
        }

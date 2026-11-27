# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Helper utils for vision Mlflow models."""

import base64
import io
import ipaddress
import logging
import os
import re
import requests
import socket
import uuid

import PIL
import pandas as pd
import numpy as np
import torch

from ast import literal_eval
from PIL import Image, UnidentifiedImageError
from requests.adapters import HTTPAdapter
from typing import Callable, List, Union
from urllib.parse import urljoin, urlparse
from urllib3.connection import HTTPConnection, HTTPSConnection
from urllib3.connectionpool import HTTPConnectionPool, HTTPSConnectionPool
from urllib3.exceptions import ConnectTimeoutError, NewConnectionError


logger = logging.getLogger(__name__)

# Uncomment the following line for mlflow debug mode
# logging.getLogger("mlflow").setLevel(logging.DEBUG)

# Image urls come from untrusted inference input, so images are only fetched from public addresses.
# Deployments that need images from private endpoints can allow specific CIDR ranges with this variable.
IMAGE_URL_ALLOWED_NETWORKS_ENV_VAR = "AZUREML_IMAGE_URL_ALLOWED_NETWORKS"
_IMAGE_URL_SCHEMES = ("http", "https")
_IMAGE_URL_TIMEOUT_SECONDS = (10, 30)  # (connect, read)
_IMAGE_URL_MAX_REDIRECTS = requests.models.DEFAULT_REDIRECT_LIMIT  # 30, the limit requests applies
_IMAGE_URL_MAX_BYTES = 50 * 1024 * 1024
# Never fetched, even when allowlisted: Azure platform endpoints (WireServer / host agent), and IPv6 prefixes that
# embed an IPv4 destination this check does not decode (local-use NAT64, Teredo, and the deprecated IPv4-compatible
# and IPv4-translated forms).
_ALWAYS_BLOCKED_NETWORKS = (
    ipaddress.ip_network("168.63.129.16/32"),
    ipaddress.ip_network("64:ff9b:1::/48"),
    ipaddress.ip_network("2001::/32"),
    ipaddress.ip_network("::/96"),
    ipaddress.ip_network("::ffff:0:0:0/96"),
)
# Treated as non-public on every Python release: ranges that ipaddress classifies differently across patch releases
# (the IETF protocol assignment blocks 192.0.0.0/24 and 2001::/23, and the 3fff::/20 documentation block), and the
# deprecated 6to4 relay anycast block 192.88.99.0/24, which ipaddress reports as global.
_NON_PUBLIC_NETWORKS = (
    ipaddress.ip_network("192.0.0.0/24"),
    ipaddress.ip_network("192.88.99.0/24"),
    ipaddress.ip_network("2001::/23"),
    ipaddress.ip_network("3fff::/20"),
)
_IPV4_MAPPED_NETWORK = ipaddress.ip_network("::ffff:0:0/96")
_NAT64_NETWORK = ipaddress.ip_network("64:ff9b::/96")


def save_image(output_folder: str, img: PIL.Image.Image, format: str) -> str:
    """
    Save image in a folder designated for batch output and return image file path.

    :param output_folder: directory path where we need to save files
    :type output_folder: str
    :param img: image object
    :type img: PIL.Image.Image
    :param format: format to save image
    :type format: str
    :return: file name of image.
    :rtype: str
    """
    filename = f"image_{uuid.uuid4()}.{format.lower()}"
    img.save(os.path.join(output_folder, filename), format=format)
    return filename


def get_pil_image(image: bytes) -> PIL.Image.Image:
    """
    Convert image bytes to PIL image.

    :param image: image bytes
    :type image: bytes
    :return: PIL image object
    :rtype: PIL.Image.Image
    """
    try:
        return Image.open(io.BytesIO(image))
    except UnidentifiedImageError as e:
        logger.error("Invalid image format. Please use base64 encoding for input images.")
        raise e


def image_to_base64(img: PIL.Image.Image, format: str) -> str:
    """
    Convert image into Base64 encoded string.

    :param img: image object
    :type img: PIL.Image.Image
    :param format: image format
    :type format: str
    :return: base64 encoded string
    :rtype: str
    """
    buffered = io.BytesIO()
    img.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


def process_image(image: Union[str, bytes]) -> bytes:
    """Process image.

    If input image is in bytes format, return it as it is.
    If input image is in base64 string format, decode it to bytes.
    If input image is in url format, download it and return bytes. Only http(s) urls whose host
    resolves to public addresses (or to networks allowed via AZUREML_IMAGE_URL_ALLOWED_NETWORKS) are fetched.
    https://github.com/mlflow/mlflow/blob/master/examples/flower_classifier/image_pyfunc.py

    :param image: image in base64 string format or url or bytes.
    :type image: string or bytes
    :return: decoded image.
    :rtype: bytes
    """
    if isinstance(image, bytes):
        return image
    elif isinstance(image, str):
        if _is_valid_url(image):
            try:
                return _download_image(image)
            except requests.exceptions.RequestException as ex:
                raise ValueError(f"Unable to retrieve image from url string due to exception: {ex}")
        else:
            try:
                return base64.b64decode(image)
            except ValueError:
                raise ValueError(
                    "The provided image string cannot be decoded. " "Expected format is base64 string or url string."
                )
    else:
        raise ValueError(
            f"Image received in {type(image)} format which is not supported. "
            "Expected format is bytes, base64 string or url string."
        )


def process_image_pandas_series(image_pandas_series: pd.Series) -> pd.Series:
    """Process image in Pandas series form.

    If input image is in bytes format, return it as it is.
    If input image is in base64 string format, decode it to bytes.
    If input image is in url format, download it and return bytes.
    https://github.com/mlflow/mlflow/blob/master/examples/flower_classifier/image_pyfunc.py

    :param img: pandas series with image in base64 string format or url or bytes.
    :type img: pd.Series
    :return: decoded image in pandas series format.
    :rtype: Pandas Series
    """
    image = image_pandas_series[0]
    return pd.Series(process_image(image))


def _is_valid_url(text: str) -> bool:
    """Check if text is url or base64 string.

    :param text: text to validate
    :type text: str
    :return: True if url else false
    :rtype: bool
    """
    regex = (
        "((http|https)://)(www.)?"
        + "[a-zA-Z0-9@:%._\\+~#?&//=\\-]"
        + "{2,256}\\.[a-z]"
        + "{2,6}\\b([-a-zA-Z0-9@:%"
        + "._\\+~#?&//=]*)"
    )
    p = re.compile(regex)

    # If the string is empty
    # return false
    if str is None:
        return False

    # Return if the string
    # matched the ReGex
    if re.search(p, text):
        return True
    else:
        return False


class UnsafeImageUrlError(ValueError):
    """Raised when an image url points to a destination that is not allowed."""


def _get_allowed_networks() -> list:
    """Return the non-public networks that the deployment explicitly allows image urls to resolve to.

    IPv4-mapped addresses are checked as the IPv4 address they map to, so ranges written in IPv4-mapped IPv6
    notation are converted to IPv4, and other IPv6 ranges never match IPv4 destinations.

    :return: allowed networks
    :rtype: list
    """
    networks = []
    for entry in os.environ.get(IMAGE_URL_ALLOWED_NETWORKS_ENV_VAR, "").split(","):
        entry = entry.strip()
        if entry:
            try:
                network = ipaddress.ip_network(entry, strict=False)
            except ValueError:
                raise ValueError(f"Invalid CIDR range '{entry}' in {IMAGE_URL_ALLOWED_NETWORKS_ENV_VAR}.")
            if network.version == 6 and network.subnet_of(_IPV4_MAPPED_NETWORK):
                network = ipaddress.ip_network((network.network_address.ipv4_mapped, network.prefixlen - 96))
            networks.append(network)
    return networks


def _is_allowed_address(address: str, allowed_networks: list) -> bool:
    """Check whether images may be fetched from an ip address.

    :param address: resolved ip address
    :type address: str
    :param allowed_networks: non-public networks explicitly allowed by the deployment
    :type allowed_networks: list
    :return: True if the address is allowed, else False
    :rtype: bool
    """
    ip = ipaddress.ip_address(address.split("%", 1)[0])
    if ip.version == 6 and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    # 6to4 and NAT64 addresses reach the IPv4 address they embed, so that address decides whether they are public.
    destination = ip
    if ip.version == 6:
        if ip.sixtofour is not None:
            destination = ip.sixtofour
        elif ip in _NAT64_NETWORK:
            destination = ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)
    for candidate in (ip, destination):
        if (candidate.is_loopback or candidate.is_link_local or candidate.is_multicast or candidate.is_unspecified
                or any(candidate in network for network in _ALWAYS_BLOCKED_NETWORKS)):
            return False
    if any(ip in network for network in allowed_networks):
        return True
    return (
        destination.is_global
        and not destination.is_reserved
        and not (destination.version == 6 and destination.is_site_local)
        and not any(destination in network for network in _NON_PUBLIC_NETWORKS)
    )


def _resolve_allowed_addresses(host: str, port: int) -> List[str]:
    """Resolve an image url host, ensuring that every address it resolves to is allowed.

    :param host: host name or ip address
    :type host: str
    :param port: port number
    :type port: int
    :return: resolved addresses
    :rtype: List[str]
    """
    allowed_networks = _get_allowed_networks()
    try:
        address_infos = socket.getaddrinfo(host, port, 0, socket.SOCK_STREAM)
    except (socket.gaierror, UnicodeError) as ex:
        raise ValueError(f"Unable to resolve image url host '{host}': {ex}")
    addresses = []
    for address_info in address_infos:
        address = address_info[4][0]
        if not _is_allowed_address(address, allowed_networks):
            raise UnsafeImageUrlError(
                f"Image url host '{host}' resolves to a non-public address, which is not allowed. To fetch images "
                f"from private networks, set {IMAGE_URL_ALLOWED_NETWORKS_ENV_VAR} to a comma-separated list of "
                "allowed CIDR ranges."
            )
        if address not in addresses:
            addresses.append(address)
    if not addresses:
        raise ValueError(f"Unable to resolve image url host '{host}'.")
    return addresses


def _connect_to_allowed_address(connection: HTTPConnection, connect: Callable) -> socket.socket:
    """Open a urllib3 connection to an allowed address of its host.

    The host is resolved and validated again at connect time, so the connection is pinned to the validated
    addresses and cannot be redirected to another address by DNS rebinding.

    :param connection: urllib3 connection being opened
    :type connection: HTTPConnection
    :param connect: urllib3's implementation for opening the connection socket
    :type connect: Callable
    :return: connected socket
    :rtype: socket.socket
    """
    host = connection._dns_host
    addresses = _resolve_allowed_addresses(host.strip("[]"), connection.port)
    error = None
    for address in addresses:
        connection._dns_host = address
        try:
            return connect()
        except (NewConnectionError, ConnectTimeoutError) as ex:
            error = ex
        finally:
            connection._dns_host = host
    raise error


class _AllowedAddressHTTPConnection(HTTPConnection):
    """urllib3 HTTP connection that only connects to allowed addresses."""

    def _new_conn(self) -> socket.socket:
        return _connect_to_allowed_address(self, super()._new_conn)


class _AllowedAddressHTTPSConnection(HTTPSConnection):
    """urllib3 HTTPS connection that only connects to allowed addresses."""

    def _new_conn(self) -> socket.socket:
        return _connect_to_allowed_address(self, super()._new_conn)


class _AllowedAddressHTTPConnectionPool(HTTPConnectionPool):
    """urllib3 HTTP connection pool that only connects to allowed addresses."""

    ConnectionCls = _AllowedAddressHTTPConnection


class _AllowedAddressHTTPSConnectionPool(HTTPSConnectionPool):
    """urllib3 HTTPS connection pool that only connects to allowed addresses."""

    ConnectionCls = _AllowedAddressHTTPSConnection


class _AllowedAddressHTTPAdapter(HTTPAdapter):
    """Requests adapter whose connections are pinned to allowed addresses.

    Urls that would be fetched through a proxy are rejected by _validate_image_url, since a proxy resolves
    the host itself and the destination could not be pinned.
    """

    def init_poolmanager(self, *args, **kwargs):
        """Initialize the pool manager with connection pools that validate addresses at connect time."""
        super().init_poolmanager(*args, **kwargs)
        self.poolmanager.pool_classes_by_scheme = {
            "http": _AllowedAddressHTTPConnectionPool,
            "https": _AllowedAddressHTTPSConnectionPool,
        }


class _NoAutoRedirectSession(requests.Session):
    """Requests session that never processes redirects itself.

    Requests drains the body of a redirect response, even with allow_redirects=False, to prepare
    Response.next. Redirects are followed by _download_image instead, so redirect bodies are never read.
    """

    def resolve_redirects(self, *args, **kwargs):
        """Do not prepare or follow redirects."""
        return iter(())


class _PreparedUrlAuth(requests.auth.AuthBase):
    """Apply .netrc or url credentials for the host that requests connects to.

    requests looks up .netrc credentials for the url before preparing it, which can name a different host than the
    one it connects to. Passing this explicit auth disables that lookup. As in requests' redirect handling, .netrc
    credentials replace a carried Authorization header, and url credentials are only used for the original url.
    """

    def __init__(self, use_url_credentials: bool):
        """Initialize the auth.

        :param use_url_credentials: whether to apply credentials embedded in the url
        :type use_url_credentials: bool
        """
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


def _validate_image_url(url: str) -> None:
    """Validate that an image url uses http(s), is not fetched through a proxy and only resolves to allowed addresses.

    :param url: image url
    :type url: str
    """
    parsed = urlparse(url)
    if parsed.scheme not in _IMAGE_URL_SCHEMES or not parsed.hostname:
        raise UnsafeImageUrlError("Only http and https image urls with a host name are supported.")
    if requests.utils.select_proxy(url, requests.utils.get_environ_proxies(url)):
        raise UnsafeImageUrlError(
            "Image urls cannot be downloaded through a proxy. Send the image as a base64 string, or exclude the "
            "image host from the proxy with NO_PROXY."
        )
    _resolve_allowed_addresses(parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80))


def _prepare_url(url: str) -> str:
    """Return a url as requests sends it.

    requests and urlparse can disagree on the host of an unusual url (for example, one with a backslash in the
    authority), so destination, proxy and origin checks use the prepared url.

    :param url: url
    :type url: str
    :return: prepared url
    :rtype: str
    """
    request = requests.PreparedRequest()
    request.prepare_url(url, None)
    return request.url


def _read_image_content(response: requests.Response) -> bytes:
    """Read an image response body, enforcing the maximum image size.

    The limit applies to the decoded body as it is read. Content-Length is not used, since it is the size of the
    encoded body and is ignored for chunked responses.

    :param response: streamed response
    :type response: requests.Response
    :return: response body
    :rtype: bytes
    """
    content = bytearray()
    for chunk in response.iter_content(chunk_size=64 * 1024):
        content.extend(chunk)
        if len(content) > _IMAGE_URL_MAX_BYTES:
            raise ValueError(f"Image downloaded from url exceeds the maximum size of {_IMAGE_URL_MAX_BYTES} bytes.")
    return bytes(content)


def _download_image(url: str) -> bytes:
    """Download an image from a url provided in the inference input.

    Every redirect hop is validated before it is requested, and connections are pinned to validated addresses.

    :param url: image url
    :type url: str
    :return: image bytes
    :rtype: bytes
    """
    headers = {}
    with _NoAutoRedirectSession() as session:
        adapter = _AllowedAddressHTTPAdapter()
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        for hop in range(_IMAGE_URL_MAX_REDIRECTS + 1):
            prepared_url = _prepare_url(url)
            _validate_image_url(prepared_url)
            auth = _PreparedUrlAuth(use_url_credentials=hop == 0)
            with session.get(url, headers=headers, auth=auth, allow_redirects=False, stream=True,
                             timeout=_IMAGE_URL_TIMEOUT_SECONDS) as response:
                # As in requests, a redirect response without a Location target is the final response.
                redirect_target = session.get_redirect_target(response)
                if redirect_target:
                    redirect_url = urljoin(prepared_url, redirect_target)
                    # Keep credentials only where requests' own redirect handling would keep them.
                    authorization = response.request.headers.get("Authorization")
                    keep_authorization = authorization and not session.should_strip_auth(
                        prepared_url, _prepare_url(redirect_url))
                    headers = {"Authorization": authorization} if keep_authorization else {}
                    url = redirect_url
                    continue
                response.raise_for_status()  # Raise exception in case of unsuccessful response code.
                return _read_image_content(response)
    raise ValueError(f"Image url exceeded the maximum of {_IMAGE_URL_MAX_REDIRECTS} redirects.")


def get_current_device() -> torch.device:
    """Get current cuda device.

    :return: current device
    :rtype: torch.device
    """
    # check if GPU is available
    if torch.cuda.is_available():
        try:
            # get the current device index
            device_idx = torch.distributed.get_rank()
        except RuntimeError as ex:
            if "Default process group has not been initialized".lower() in str(ex).lower():
                device_idx = 0
            else:
                logger.error(str(ex))
                raise ex
        return torch.device(type="cuda", index=device_idx)
    else:
        return torch.device(type="cpu")


def string_to_nested_float_list(input_str: str) -> list:
    """Convert string to nested list of floats.

    :return: string converted to nested list of floats
    :rtype: list
    """
    if input_str in ["null", "None", "", "nan", "NoneType", np.nan, None]:
        return None
    try:
        # Use ast.literal_eval to safely evaluate the string into a list
        nested_list = literal_eval(input_str)

        # Recursive function to convert all numbers in the nested list to floats
        def to_floats(lst) -> list:
            """
            Recursively convert all numbers in a nested list to floats.

            :param lst: nested list
            :type lst: list
            :return: nested list of floats
            :rtype: list
            """
            return [to_floats(item) if isinstance(item, list) else float(item) for item in lst]

        # Use the recursive function to process the nested list
        return to_floats(nested_list)
    except (ValueError, SyntaxError) as e:
        # In case of an error during conversion, print an error message
        print(f"Invalid input {input_str}: {e}, ignoring.")
        return None


def bool_array_to_pil_image(bool_array: np.ndarray) -> PIL.Image.Image:
    """Convert boolean array to PIL Image.

    :param bool_array: boolean array
    :type bool_array: np.array
    :return: PIL Image
    :rtype: PIL.Image.Image
    """
    # Convert boolean array to uint8
    uint8_array = bool_array.astype(np.uint8) * 255

    # Create a PIL Image
    pil_image = Image.fromarray(uint8_array)

    return pil_image

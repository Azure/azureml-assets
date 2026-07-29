# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Vesta image encoder."""

import base64
import ipaddress
import os
import socket
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

from ...telemetry import logging_utils as lu
from .request_modifier import RequestModificationException


class ImageEncoder():
    """Vesta image encoder."""

    IMAGE_URL = "ImageUrl!"
    IMAGE_FILE = "ImageFile!"

    # SSRF (CWE-918) mitigation: only fetch http(s) urls that resolve to public addresses.
    ALLOWED_URL_SCHEMES = ("http", "https")
    URL_REQUEST_TIMEOUT_SECONDS = 30

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
        metadata endpoint at 169.254.169.254), reserved, multicast, or
        unspecified addresses are rejected before any network call is made.
        """
        parsed = urlparse(url)

        if parsed.scheme not in ImageEncoder.ALLOWED_URL_SCHEMES:
            raise UnsafeUrl(url)

        hostname = parsed.hostname
        if not hostname:
            raise UnsafeUrl(url)

        try:
            address_infos = socket.getaddrinfo(hostname, parsed.port, proto=socket.IPPROTO_TCP)
        except socket.gaierror as e:
            raise UnsafeUrl(url) from e

        if not address_infos:
            raise UnsafeUrl(url)

        for address_info in address_infos:
            ip = ipaddress.ip_address(address_info[4][0])
            if (
                ip.is_private
                or ip.is_loopback
                or ip.is_link_local
                or ip.is_reserved
                or ip.is_multicast
                or ip.is_unspecified
            ):
                raise UnsafeUrl(url)

    def _b64_from_url(self, url: str) -> str:
        self._validate_url(url)

        start = time.time()
        resp = requests.get(url, timeout=ImageEncoder.URL_REQUEST_TIMEOUT_SECONDS)
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

import base64
import os
import time
from pathlib import Path

import requests

from ...telemetry import logging_utils as lu
from .request_modifier import RequestModificationException


class ImageEncoder():
    IMAGE_URL = "ImageUrl!"
    IMAGE_FILE = "ImageFile!"

    def __init__(self, image_input_folder_str: str = None) -> None:
        self.__image_input_folder_str: str = None
        if image_input_folder_str:
            self.__image_input_folder_str = str(Path(image_input_folder_str))

    def encode_b64(self, image_data: str) -> str:
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

    def _b64_from_url(self, url: str) -> str:
        start = time.time()
        resp = requests.get(url)
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
    def __init__(self, *args: object) -> None:
        super().__init__(f"{ImageEncoder.IMAGE_URL} used in data, but url did not respond succesfully.")


class FolderNotMounted(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(f"{ImageEncoder.IMAGE_FILE} used in data, but no folder is mounted.")


class VestaImageModificationException(RequestModificationException):
    def __init__(self) -> None:
        super().__init__("The ImageEncoder raised an exception")

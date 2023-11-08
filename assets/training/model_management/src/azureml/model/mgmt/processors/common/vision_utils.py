# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Helper utils for vision Mlflow models."""

import logging
import os
import tempfile
import uuid
import PIL
import pandas as pd
import base64
import io
import re
import requests
import torch
from ast import literal_eval
import numpy as np

from PIL import Image, UnidentifiedImageError
from typing import Tuple, Union


logger = logging.getLogger(__name__)

# Uncomment the following line for mlflow debug mode
# logging.getLogger("mlflow").setLevel(logging.DEBUG)


def create_temp_file(request_body: bytes, parent_dir: str) -> Tuple[str, Image.Image]:
    """Create temporory file from image bytes, save image and return path to the file and PIL Image.

    :param request_body: Image
    :type request_body: bytes
    :param parent_dir: directory name
    :type parent_dir: str
    :return: Path to the file, PIL Image
    :rtype: Tuple[str, Image.Image]
    """
    with tempfile.NamedTemporaryFile(dir=parent_dir, mode="wb", delete=False) as image_file_fp:
        img_path = image_file_fp.name + ".png"
        img = get_pil_image(request_body)
        img.save(img_path)
        return img_path, img


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
    If input image is in url format, download it and return bytes.
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
                response = requests.get(image)
                response.raise_for_status()  # Raise exception in case of unsuccessful response code.
                image = response.content
                return image
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


def string_to_nested_float_list(s: str) -> list:
    """Convert string to nested list of floats.

    :return: string converted to nested list of floats
    :rtype: list
    """
    # Check if string is None or empty
    if s in ["null", "None", "", "nan", "NoneType", np.nan, None]:
        return None
    # Check if string matches the expected format using a regex pattern.
    # This pattern ensures that the string only contains brackets, numbers, dots, and commas.
    if not re.match(r"^\[((\[|\]|\s|\d|\,|\.|-)*?)\]$", s):
        raise ValueError("Invalid input format, please use a nested list of ints or floats.")

    # Convert string to nested list using literal_eval
    nested_list = literal_eval(s)

    # Helper function to convert nested lists to floats
    def to_float_recursive(lst)-> list:
        """
        Recursively convert nested lists to floats.
        :param lst: nested list
        :type lst: list
        :return: nested list of floats
        :rtype: list
        """
        for i, item in enumerate(lst):
            if isinstance(item, list):
                to_float_recursive(item)
            else:
                lst[i] = float(item)

    to_float_recursive(nested_list)
    return nested_list


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

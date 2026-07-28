# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Helper utils for vision Mlflow models."""

import logging
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
from typing import Union


logger = logging.getLogger(__name__)

# Uncomment the following line for mlflow debug mode
# logging.getLogger("mlflow").setLevel(logging.DEBUG)


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

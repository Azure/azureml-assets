# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Utility methods for StableDiffusion."""

import base64
from io import BytesIO
import io
import logging
import os
import re
import uuid
import pandas as pd
import PIL

import requests
import torch

from PIL import Image, UnidentifiedImageError
from config import DatatypeLiterals


logger = logging.getLogger(__name__)


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


def image_to_str(img: PIL.Image.Image) -> str:
    """
    Convert image into Base64 encoded string.

    :param img: image object
    :type img: PIL.Image.Image
    :return: base64 encoded string
    :rtype: str
    """
    buffered = BytesIO()
    img.save(buffered, format=DatatypeLiterals.IMAGE_FORMAT)
    img_str = base64.b64encode(buffered.getvalue()).decode(DatatypeLiterals.STR_ENCODING)
    return img_str


def save_image(output_folder: str, img: PIL.Image.Image) -> str:
    """
    Save image in a folder designated for batch output and return image file path.

    :param output_folder: directory path where we need to save files
    :type output_folder: str
    :param img: image object
    :type img: PIL.Image.Image
    :return: file name of image.
    :rtype: str
    """
    filename = f"image_{uuid.uuid4()}.{DatatypeLiterals.IMAGE_FORMAT.lower()}"
    img.save(os.path.join(output_folder, filename), format=DatatypeLiterals.IMAGE_FORMAT)
    return filename


def process_image(img: pd.Series) -> pd.Series:
    """Process input image and return bytes.

    If input image is in base64 string format, decode it to bytes.
    If input image is in url format, download it and return bytes.
    https://github.com/mlflow/mlflow/blob/master/examples/flower_classifier/image_pyfunc.py

    :param img: pandas series with image in base64 string format or url.
    :type img: pd.Series
    :return: decoded image in pandas series format.
    :rtype: Pandas Series
    """
    image = img[0]
    if isinstance(image, bytes):
        return img
    elif isinstance(image, str):
        if _is_valid_url(image):
            image = requests.get(image).content
            return pd.Series(image)
        else:
            try:
                return pd.Series(base64.b64decode(image))
            except ValueError:
                raise ValueError(
                    "The provided image string cannot be decoded." "Expected format is base64 string or url string."
                )
    else:
        raise ValueError(
            f"Image received in {type(image)} format which is not supported."
            "Expected format is bytes, base64 string or url string."
        )


def _is_valid_url(text: str) -> bool:
    """Check if text is url or base64 string.

    :param text: text to validate
    :type text: str
    :return: True if url else false
    :rtype: bool
    """
    regex = (
        "((http|https)://)(www.)?"
        + "[a-zA-Z0-9@:%._\\+~#?&//=]"
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

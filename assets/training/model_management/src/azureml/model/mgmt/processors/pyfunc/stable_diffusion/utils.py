# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Utility methods for StableDiffusion."""

import base64
from io import BytesIO
import os
import PIL
import uuid

from constants import DatatypeLiterals


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
    :return: Path where image is saved/persisted.
    :rtype: str
    """
    filename = f'image_{uuid.uuid4()}.jpeg'
    filename = os.path.join(output_folder, filename)
    img.save(filename, format=DatatypeLiterals.IMAGE_FORMAT)

    return filename

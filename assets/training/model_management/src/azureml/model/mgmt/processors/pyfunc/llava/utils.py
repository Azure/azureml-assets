import base64
import re
import requests

from typing import Union


def process_image(image: Union[str, bytes]) -> bytes:
    """Process input image.

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
            return requests.get(image).content
        else:
            try:
                return base64.b64decode(image)
            except ValueError:
                raise ValueError(
                    "The provided image string cannot be decoded. Expected format is Base64 or URL string."
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

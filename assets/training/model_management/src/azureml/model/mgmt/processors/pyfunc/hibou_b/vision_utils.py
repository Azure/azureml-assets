import os
import torch
import requests
import base64
from PIL import Image
import io


def get_current_device():
    """Return the current device (CUDA if available, otherwise CPU)."""
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def process_image(image_input):
    """
    Process the image input and return bytes.

    If the input is:
      - a URL (starting with 'http'): download the image bytes.
      - a file path: open and read the file.
      - a base64-encoded string: decode it.
      - bytes: return as-is.
    """
    # If it's a URL, download the content.
    if isinstance(image_input, str):
        if image_input.startswith("http"):
            response = requests.get(image_input)
            if response.status_code != 200:
                raise ValueError(
                    f"Failed to download image. Status code: {response.status_code}"
                )
            return response.content
        # Check if it's a file path.
        elif os.path.exists(image_input):
            with open(image_input, "rb") as f:
                return f.read()
        else:
            # Attempt to decode as base64.
            try:
                return base64.b64decode(image_input)
            except Exception as e:
                raise ValueError(
                    "Invalid image input string; not a URL, file path, or valid base64 data."
                ) from e
    elif isinstance(image_input, bytes):
        return image_input
    else:
        raise ValueError("Unsupported image input type; expected a string or bytes.")

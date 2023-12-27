# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Output schema of the models."""


from dataclasses import dataclass
from typing import Optional, Union
from utils import image_to_base64, box_logger


@dataclass
class TextToImageSchema:
    """Text to image task schema."""

    generated_image: str
    prompt: str
    nsfw_content_detected: Optional[bool] = None

    def __post_init__(self):
        """Convert the generated image object to base64 format."""
        self.generated_image = image_to_base64(self.generated_image, "PNG")


@dataclass
class ImageTaskInferenceResult:
    """Image task inference result of a single request."""

    response: Union[TextToImageSchema, None]
    inference_time_ms: float
    error: Optional[str] = None

    def print_results(self):
        """Print the inference results of a single prompt."""
        if self.error:
            msg = f"## Inference Results ##\n ERROR: {self.error}"
            box_logger(msg)

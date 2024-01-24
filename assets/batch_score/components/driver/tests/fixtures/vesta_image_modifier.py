# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains fixtures to mock vesta image modifier."""

import pytest

from src.batch_score.common.request_modification.modifiers.vesta_chat_completion_image_modifier import (
    VestaChatCompletionImageModifier,
)
from src.batch_score.common.request_modification.modifiers.vesta_image_encoder import (
    ImageEncoder,
)
from src.batch_score.common.request_modification.modifiers.vesta_image_modifier import (
    VestaImageModifier,
)

MOCKED_IMAGE_ENCODING = "MOCKED_ENCODING"
MOCKED_BINARY_FROM_FILE = "MOCKED_BINARY_FROM_FILE"
MOCKED_BINARY_FROM_URL = "MOCKED_BINARY_FROM_URL"
IMAGE_ENCODER_NAMESPACE = "src.batch_score.common.request_modification.modifiers.vesta_image_encoder.ImageEncoder"


@pytest.fixture
def make_vesta_image_modifier(make_image_encoder):
    """Mock vesta image modifier."""
    def make(image_encoder: ImageEncoder = make_image_encoder()):
        """Make a mock vesta image modifier."""
        return VestaImageModifier(
            image_encoder=image_encoder
        )

    return make


@pytest.fixture
def make_vesta_chat_completion_image_modifier(make_image_encoder):
    """Mock vesta chat completion image modifier."""
    def make(image_encoder: ImageEncoder = make_image_encoder()):
        """Make a mock vesta chat completion image modifier."""
        return VestaChatCompletionImageModifier(
            image_encoder=image_encoder
        )

    return make


@pytest.fixture
def make_image_encoder():
    """Mock image encoder."""
    def make(image_input_folder_str: str = None):
        return ImageEncoder(
            image_input_folder_str=image_input_folder_str
        )

    return make


@pytest.fixture
def mock_encode_b64(monkeypatch):
    """Mock encode b64."""
    state = {"exception": None}

    def _encode_b64(self, image_data) -> str:
        if state["exception"]:
            raise state["exception"]

        return MOCKED_IMAGE_ENCODING

    monkeypatch.setattr(f"{IMAGE_ENCODER_NAMESPACE}.encode_b64", _encode_b64)
    return state


@pytest.fixture
def mock__b64_from_file(monkeypatch):
    """Mock b64 from file."""
    requested_files = []

    def __b64_from_file(self, path: str):
        requested_files.append(path)
        return MOCKED_BINARY_FROM_FILE

    monkeypatch.setattr(f"{IMAGE_ENCODER_NAMESPACE}._b64_from_file", __b64_from_file)
    return requested_files


@pytest.fixture
def mock__b64_from_url(monkeypatch):
    """Mock b64 from url."""
    requested_urls = []

    def __b64_from_url(self, url: str):
        requested_urls.append(url)
        return MOCKED_BINARY_FROM_URL

    monkeypatch.setattr(f"{IMAGE_ENCODER_NAMESPACE}._b64_from_url", __b64_from_url)
    return requested_urls

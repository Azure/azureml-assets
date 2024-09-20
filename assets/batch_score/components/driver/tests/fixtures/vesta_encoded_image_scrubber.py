# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains fixtures to mock vesta encoded image scrubber."""

import pytest

from src.batch_score.common.request_modification.modifiers.vesta_chat_completion_encoded_image_scrubber import (
    VestaChatCompletionEncodedImageScrubber,
)
from src.batch_score.common.request_modification.modifiers.vesta_encoded_image_scrubber import (
    VestaEncodedImageScrubber,
)


@pytest.fixture()
def make_vesta_encoded_image_scrubber():
    """Mock vesta encoded image scrubber."""
    def make():
        """Make a mock vesta encoded image scrubber."""
        return VestaEncodedImageScrubber()

    return make


@pytest.fixture()
def make_vesta_chat_completion_encoded_image_scrubber():
    """Mock vesta chat completion encoded image scrubber."""
    def make():
        """Make a mock vesta chat completion encoded image scrubber."""
        return VestaChatCompletionEncodedImageScrubber()

    return make

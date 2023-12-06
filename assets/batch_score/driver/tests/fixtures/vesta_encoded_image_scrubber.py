# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pytest

from src.batch_score.common.request_modification.modifiers.vesta_chat_completion_encoded_image_scrubber import (
    VestaChatCompletionEncodedImageScrubber,
)
from src.batch_score.common.request_modification.modifiers.vesta_encoded_image_scrubber import (
    VestaEncodedImageScrubber,
)


@pytest.fixture()
def make_vesta_encoded_image_scrubber():
    def make():
        return VestaEncodedImageScrubber()
    
    return make

@pytest.fixture()
def make_vesta_chat_completion_encoded_image_scrubber():
    def make():
        return VestaChatCompletionEncodedImageScrubber()
    
    return make

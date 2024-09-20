# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains unit tests for vesta chat completion encoded image scrubber."""

import pytest

from src.batch_score.common.request_modification.modifiers.vesta_chat_completion_encoded_image_scrubber import (
    VestaChatCompletionEncodedImageScrubber,
)


def test_image_scrubber(make_vesta_chat_completion_encoded_image_scrubber):
    """Test image scrubber."""
    scrubber: VestaChatCompletionEncodedImageScrubber = make_vesta_chat_completion_encoded_image_scrubber()

    request_obj = {"non-vesta-obj": "val"}
    with pytest.raises(Exception):
        scrubber.modify(request_obj=request_obj)

    request_obj = {"messages": [{"role": "user", "content": ["Review the image.", {"image": "some-encoded-image"}]}]}
    modified_obj = scrubber.modify(request_obj=request_obj)
    assert modified_obj["messages"][0]["content"][1]["image"] == "<Encoded image data has been scrubbed>"

    image_url_data = "ImageUrl!some-url"
    request_obj = {"messages": [{"role": "user", "content": ["Review the image.", {"image": image_url_data}]}]}
    modified_obj = scrubber.modify(request_obj=request_obj)
    assert modified_obj["messages"][0]["content"][1]["image"] == image_url_data

    image_file_data = "ImageFile!some-file-path"
    request_obj = {"messages": [{"role": "user", "content": ["Review the image.", {"image": image_file_data}]}]}
    modified_obj = scrubber.modify(request_obj=request_obj)
    assert modified_obj["messages"][0]["content"][1]["image"] == image_file_data

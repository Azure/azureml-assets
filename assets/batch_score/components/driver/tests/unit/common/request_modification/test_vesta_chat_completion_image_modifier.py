# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains unit tests for vesta chat completion image modifier."""

import pytest
from urllib.parse import urlparse

from src.batch_score.common.request_modification.modifiers.vesta_chat_completion_image_modifier import (
    VestaChatCompletionImageModifier,
)
from src.batch_score.common.request_modification.modifiers.vesta_image_encoder import (
    FolderNotMounted,
    VestaImageModificationException,
)
from tests.fixtures.vesta_image_modifier import MOCKED_BINARY_FROM_URL


@pytest.mark.parametrize("expected_result, request_obj", [
    [True, {"messages": [{"role": "user",
                          "content": [{"image": "base64encoded"}, "Transcribe this image please"]}]}],
    [False, {"messages": [{"role": "user",
                           "content": [{"image": "base64encoded"}, "Transcribe this image please"]},
                          {"hello": "world"}]}],
    [False, {"messages": [{"hello": "world"}]}],
    [False, {"invalid": [{"role": "user",
                          "content": [{"image": "base64encoded"}, "Transcribe this image please"]}]}]
])
def test_is_vesta_chat_completion_payload(request_obj, expected_result):
    """Test is vesta chat completion payload."""
    assert VestaChatCompletionImageModifier.is_vesta_chat_completion_payload(request_obj) == expected_result


def test_modify(mock_get_logger, make_vesta_chat_completion_image_modifier, mock__b64_from_url):
    """Test modify."""
    vesta_request_obj = {
        "messages": [
            {
                "role": "user",
                "content": [
                    "Review the images below.",
                    {
                        "image": "ImageUrl!https://fake.url"
                    },
                    "Can you tell me the colors of dots in the picture."
                ]
            },
            {
                "role": "developer",
                "content": [
                    "Review the images below.",
                    {
                        "image": "/9j/4AAQSkZJRgABAQAAAQABAAD/"
                    },
                    {
                        "image_hr": "/9j/4AAQSkZJRgABAQAAAQABAAD/"
                    },
                    "Can you tell me the colors of dots in the picture. How many dots are there?"
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "image_url":
                        {
                            "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAACAAAAAgACAIAAA…kSuQmCC",
                            "detail": "auto"
                        }
                    },
                    "Describe the image."
                ]
            }
        ]}
    vesta_image_modifier: VestaChatCompletionImageModifier = make_vesta_chat_completion_image_modifier()

    modified_request_obj = vesta_image_modifier.modify(request_obj=vesta_request_obj)

    # Assert the two URLs were called
    assert any("fake.url" in urlparse(url).hostname for url in mock__b64_from_url)

    # Assert modifications are correct
    # raise Exception(str(modified_request_obj["messages"][1]["content"][2]))
    assert modified_request_obj["messages"][0]["content"][1]["image"] == MOCKED_BINARY_FROM_URL
    assert modified_request_obj["messages"][1]["content"][1]["image"] == "/9j/4AAQSkZJRgABAQAAAQABAAD/"
    assert modified_request_obj["messages"][1]["content"][2]["image_hr"] == "/9j/4AAQSkZJRgABAQAAAQABAAD/"
    assert modified_request_obj["messages"][2]["content"][0]["image_url"]["url"] == \
        "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAACAAAAAgACAIAAA…kSuQmCC"


def test_modify_invalid_image(mock_get_logger, make_vesta_chat_completion_image_modifier, mock_encode_b64):
    """Test modify invalid image case."""
    vesta_request_obj = {"messages": [{"role": "user", "content": [{"image": "invalid_image"}]}]}
    vesta_chat_completion_image_modifier = make_vesta_chat_completion_image_modifier()

    mock_encode_b64["exception"] = FolderNotMounted()
    with pytest.raises(VestaImageModificationException):
        vesta_chat_completion_image_modifier.modify(vesta_request_obj)

    mock_encode_b64["exception"] = Exception()
    with pytest.raises(VestaImageModificationException):
        vesta_chat_completion_image_modifier.modify(vesta_request_obj)

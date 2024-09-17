# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains unit tests for input type modifier."""

from src.batch_score_oss.common.request_modification.modifiers.input_type_modifier import InputTypeModifier
from src.batch_score_oss.common.common_enums import InputType


def test_get_input_type():
    """Test input type detection."""
    assert InputTypeModifier.get_input_type(request_obj={"messages": []}) == InputType.Unknown
    assert InputTypeModifier.get_input_type(request_obj={"no_messages": []}) == InputType.Unknown
    assert InputTypeModifier.get_input_type(request_obj={
        "messages": [{"content": "just text"}, {"content": [{"type": "text"}]}]
    }) == InputType.TextOnly
    assert InputTypeModifier.get_input_type(request_obj={
        "messages": [
            {"content": [{"type": "image"}, "just text"]},
            {"content": [{"type": "text"}]}
        ]
    }) == InputType.ImageAndText
    assert InputTypeModifier.get_input_type(request_obj={
        "messages": [
            {"content": "just text"}, 
            {"content": [{"type": "text"}]}, 
            {"content": [{"type": "image"}]}
        ]
    }) == InputType.ImageAndText
    assert InputTypeModifier.get_input_type(request_obj={
        "messages": [
            {"content": [{"type": "image_url"}, "just text"]},
            {"content": [{"type": "text"}]}
        ]
    }) == InputType.ImageAndText
    assert InputTypeModifier.get_input_type(request_obj={
        "messages": [
            {"content": "just text"}, 
            {"content": [{"type": "text"}]},
            {"content": [{"type": "image_url"}]}
        ]
    }) == InputType.ImageAndText
    assert InputTypeModifier.get_input_type(request_obj={
        "messages": [
            {"content": [{"type": "image"}]},
            {"content": [{"type": "image_url"}]}
        ]
    }) == InputType.Image
    assert InputTypeModifier.get_input_type(request_obj={
        "messages": [
            {"content": [{"type": "image"}, {"type": "image_url"}]}
        ]
    }) == InputType.Image


def test_modify():
    """Test modify."""
    input_request_obj = {
        "messages": [
            {
                "role": "user",
                "content": [
                    "Review the images below.",
                    {
                        "type": "image",
                        "image": "ImageUrl!https://fake.url"
                    },
                    "Can you tell me the colors of dots in the picture."
                ]
            },
            {
                "role": "developer",
                "content": [
                    {
                        "image": "/9j/4AAQSkZJRgABAQAAAQABAAD/"
                    },
                    {
                        "image_hr": "/9j/4AAQSkZJRgABAQAAAQABAAD/"
                    }
                ]
            },
        ]}
    input_type_modifier = InputTypeModifier()

    modified_request_obj = input_type_modifier.modify(request_obj=input_request_obj)

    # Assert modifications are correct
    assert modified_request_obj["input_type"] == InputType.ImageAndText


def test_modify_invalid_content():
    """Test modify unknown type."""
    input_request_obj = {"messages": [{"content": []}]}
    input_type_modifier = InputTypeModifier()

    modified_request_obj = input_type_modifier.modify(request_obj=input_request_obj)

    assert "input_type" not in modified_request_obj

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains unit tests for vesta encoded image scrubber."""

import pytest

from src.batch_score.common.request_modification.modifiers.vesta_encoded_image_scrubber import (
    VestaEncodedImageScrubber,
)


def test_image_scrubber_invalid_request_raises_exception(make_vesta_encoded_image_scrubber):
    """Test image scrubber."""
    scrubber: VestaEncodedImageScrubber = make_vesta_encoded_image_scrubber()

    request_obj = {"non-vesta-obj": "val"}
    with pytest.raises(Exception):
        scrubber.modify(request_obj=request_obj)


def test_image_scrubber_image_data_scrubbed(make_vesta_encoded_image_scrubber):
    """Test image scrubber when data should be scrubbed."""
    scrubber: VestaEncodedImageScrubber = make_vesta_encoded_image_scrubber()

    request_obj = {"transcript": [{"type": "image", "data": "some-encoded-image"}]}
    modified_obj = scrubber.modify(request_obj=request_obj)
    assert modified_obj["transcript"][0]["data"] == "<Encoded image data has been scrubbed>"


def test_image_scrubber_image_data_unchanged(make_vesta_encoded_image_scrubber):
    """Test image scrubber when data should be unchanged."""
    scrubber: VestaEncodedImageScrubber = make_vesta_encoded_image_scrubber()

    image_url_data = "ImageUrl!some-url"
    request_obj = {"transcript": [{"type": "image", "data": image_url_data}]}
    modified_obj = scrubber.modify(request_obj=request_obj)
    assert modified_obj["transcript"][0]["data"] == image_url_data

    image_file_data = "ImageFile!some-file-path"
    request_obj = {"transcript": [{"type": "image", "data": image_file_data}]}
    modified_obj = scrubber.modify(request_obj=request_obj)
    assert modified_obj["transcript"][0]["data"] == image_file_data

    image_file_data = "ImageFile!some-file-path"
    request_obj = {"transcript": [{"type": "image_hr", "data": image_file_data}]}
    modified_obj = scrubber.modify(request_obj=request_obj)
    assert modified_obj["transcript"][0]["data"] == image_file_data

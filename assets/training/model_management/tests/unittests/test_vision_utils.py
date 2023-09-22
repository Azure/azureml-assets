# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Test vision utility code."""

import base64
import pytest
import requests

import pandas as pd

from unittest.mock import patch

from azureml.model.mgmt.processors.common.vision_utils import process_image, process_image_pandas_series, _is_valid_url


class TestVisionUtilities:
    """Test vision utility functions."""

    def test_process_image_bytes(self):
        """Test process_image() with a bytes object."""
        image_bytes = bytes([0, 1])

        assert process_image(image_bytes) == image_bytes

    @patch("requests.get")
    def test_process_image_url_successful(self, mock_requests_get):
        """Test process_image() with a 'correct' URL."""
        image_url = "http://microsoft.com/image.jpg"
        image_bytes = bytes([2, 3])

        response = mock_requests_get.return_value
        response.content = image_bytes

        assert process_image(image_url) == image_bytes

    @patch("requests.get")
    def test_process_image_url_unavailable(self, mock_requests_get):
        """Test process_image() with an 'incorrect' URL."""
        image_url = "http://microsoft.com/this_image_does_not_exist.jpg"

        response = mock_requests_get.return_value
        response.raise_for_status.side_effect = requests.exceptions.RequestException("Url unavailable.")

        with pytest.raises(ValueError) as e:
            assert process_image(image_url)
        assert "Unable to retrieve image from url string due to exception: " in str(e.value)

    def test_process_image_b64(self):
        """Test process_image() with a b64encoded string."""
        image_bytes = b"45"
        image_str = base64.encodebytes(image_bytes).decode("utf-8")

        assert process_image(image_str) == image_bytes

    def test_process_image_url_invalid(self):
        """Test process_image() with an invalid URL."""
        image_url = "ftp://microsoft.com/image.jpg"

        with pytest.raises(ValueError) as e:
            assert process_image(image_url)
        assert "The provided image string cannot be decoded. Expected format is base64 string or url string." in str(
            e.value
        )

    def test_process_image_list(self):
        """Test process_image() with an invalid image type."""
        image_array = [6, 7]

        with pytest.raises(ValueError) as e:
            assert process_image(image_array)
        assert (
            "Image received in <class 'list'> format which is not supported. "
            "Expected format is bytes, base64 string or url string."
        ) in str(e.value)

    def test_process_image_pandas_series1(self):
        """Test process_image_pandas_series() with a b64encoded string."""
        image_bytes = b"89"
        image_str = base64.encodebytes(image_bytes).decode("utf-8")

        assert (process_image_pandas_series(pd.Series(image_str)) == pd.Series(image_bytes)).all()

    def test_process_image_pandas_series2(self):
        """Test process_image_pandas_series() with similar usage as in the CLIP wrapper."""
        input_data = pd.DataFrame(
            columns=["image", "text"],
            data=[
                [base64.encodebytes(b"image with things").decode("utf-8"), "thing1,thing2"],
                [base64.encodebytes(b"image with stuff").decode("utf-8"), "stuff"],
                [base64.encodebytes(b"blank image").decode("utf-8"), "nothing"],
            ]
        )

        expected_decoded_images = [b"image with things", b"image with stuff", b"blank image"]

        decoded_images = input_data.loc[:, ["image"]].apply(axis=1, func=process_image_pandas_series)
        for decoded_image, expected_decoded_image in zip(decoded_images.iloc[:, 0], expected_decoded_images):
            assert decoded_image == expected_decoded_image

    def test_is_valid_url_llava_example(self):
        """Test that the function validating the URL validates the LLaVA example URL."""
        assert _is_valid_url("https://llava-vl.github.io/static/images/view.jpg")

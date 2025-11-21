# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import unittest
from unittest.mock import patch, Mock
import os
import base64
import io

from PIL import Image

from utils import map_env_vars_to_vllm_server_kwargs, map_env_vars_to_vllm_server_args, get_gpu_device_capability, image_to_base64, convert_image_to_bytes


class TestUtils(unittest.TestCase):
    @patch.dict(os.environ, {
        "MODEL": "test_model",
        "TOKENIZER": "test_tokenizer",
        "REVISION": "test_revision",
        "TOKENIZER_REVISION": "test_tokenizer_revision",
        "TOKENIZER_MODE": "test_tokenizer_mode",
        "TRUST_REMOTE_CODE": "True",
        "NON_EXISTENT": "non_existent"
    })
    def test_map_env_vars_to_vllm_server_kwargs(self):
        expected_kwargs = {
            "model": "test_model",
            "tokenizer": "test_tokenizer",
            "revision": "test_revision",
            "tokenizer-revision": "test_tokenizer_revision",
            "tokenizer-mode": "test_tokenizer_mode",
        }
        kwargs = map_env_vars_to_vllm_server_kwargs()
        self.assertEqual(expected_kwargs, kwargs)

    @patch.dict(os.environ, {
        "MODEL": "test_model",
        "TOKENIZER": "test_tokenizer",
        "TRUST_REMOTE_CODE": "True",
        "ENABLE_LORA": "True",
        "WORKER_USE_RAY": "False",
    })
    def test_map_env_vars_to_vllm_server_args(self):
        expected_args = {"trust-remote-code", "enable-lora"}
        args = map_env_vars_to_vllm_server_args()
        self.assertEqual(expected_args, set(args))

    @patch.dict(os.environ, {
        "TRUST_REMOTE_CODE": "test_remote_code"
    })
    def test_map_env_vars_to_vllm_server_args_fails(self):
        self.assertRaises(Exception, map_env_vars_to_vllm_server_args)


def test_get_gpu_device_capability():
    MockProperties = type('MockProperties', (), {'major': 5, 'minor': 1})
    with patch('torch.cuda.get_device_properties', return_value=MockProperties) as mock_get_device_properties:
        capability = get_gpu_device_capability()
        assert isinstance(capability, float), "The function should return a float"
        assert capability == 5.1, "The capability should be 5.1"
        mock_get_device_properties.assert_called_once_with(0)


def test_image_to_base64():
    # Create a small red image
    img = Image.new('RGB', (1, 1), color='red')

    # Convert the image to base64
    img_base64 = image_to_base64(img, 'PNG')

    # Convert the base64 back to an image
    img_bytes = base64.b64decode(img_base64)
    img_file = io.BytesIO(img_bytes)
    img_new = Image.open(img_file)

    # Check that the original and final images are the same
    assert img.tobytes() == img_new.tobytes(), "The original and final images should be the same"


def test_convert_base64_image():
    # Create a base64 encoded string
    base64_image = base64.b64encode(b'Test image data').decode('utf-8')
    result = convert_image_to_bytes(base64_image)
    assert result == b'Test image data'


@patch('requests.get')
def test_convert_url_image(mock_get):
    # Mock a URL image
    url = 'http://example.com/image.jpg'
    mock_get.return_value = Mock(content=b'Test image data')
    result = convert_image_to_bytes(url)
    assert result == b'Test image data'


if __name__ == "__main__":
    unittest.main()
=======
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import unittest
import pytest
import requests
from unittest.mock import patch, Mock
import os
import base64
import io

from PIL import Image

from utils import map_env_vars_to_vllm_server_kwargs, map_env_vars_to_vllm_server_args, get_gpu_device_capability, image_to_base64, convert_image_to_bytes


class TestUtils(unittest.TestCase):
    @patch.dict(os.environ, {
        "MODEL": "test_model",
        "TOKENIZER": "test_tokenizer",
        "REVISION": "test_revision",
        "TOKENIZER_REVISION": "test_tokenizer_revision",
        "TOKENIZER_MODE": "test_tokenizer_mode",
        "TRUST_REMOTE_CODE": "True",
        "NON_EXISTENT": "non_existent"
    })
    def test_map_env_vars_to_vllm_server_kwargs(self):
        expected_kwargs = {
            "model": "test_model",
            "tokenizer": "test_tokenizer",
            "revision": "test_revision",
            "tokenizer-revision": "test_tokenizer_revision",
            "tokenizer-mode": "test_tokenizer_mode",
        }
        kwargs = map_env_vars_to_vllm_server_kwargs()
        self.assertEqual(expected_kwargs, kwargs)

    @patch.dict(os.environ, {
        "MODEL": "test_model",
        "TOKENIZER": "test_tokenizer",
        "TRUST_REMOTE_CODE": "True",
        "ENABLE_LORA": "True",
        "WORKER_USE_RAY": "False",
    })
    def test_map_env_vars_to_vllm_server_args(self):
        expected_args = {"trust-remote-code", "enable-lora"}
        args = map_env_vars_to_vllm_server_args()
        self.assertEqual(expected_args, set(args))

    @patch.dict(os.environ, {
        "TRUST_REMOTE_CODE": "test_remote_code"
    })
    def test_map_env_vars_to_vllm_server_args_fails(self):
        self.assertRaises(Exception, map_env_vars_to_vllm_server_args)


def test_get_gpu_device_capability():
    MockProperties = type('MockProperties', (), {'major': 5, 'minor': 1})
    with patch('torch.cuda.get_device_properties', return_value=MockProperties) as mock_get_device_properties:
        capability = get_gpu_device_capability()
        assert isinstance(capability, float), "The function should return a float"
        assert capability == 5.1, "The capability should be 5.1"
        mock_get_device_properties.assert_called_once_with(0)


def test_image_to_base64():
    # Create a small red image
    img = Image.new('RGB', (1, 1), color='red')

    # Convert the image to base64
    img_base64 = image_to_base64(img, 'PNG')

    # Convert the base64 back to an image
    img_bytes = base64.b64decode(img_base64)
    img_file = io.BytesIO(img_bytes)
    img_new = Image.open(img_file)

    # Check that the original and final images are the same
    assert img.tobytes() == img_new.tobytes(), "The original and final images should be the same"


def test_convert_base64_image():
    # Create a base64 encoded string
    base64_image = base64.b64encode(b'Test image data').decode('utf-8')
    result = convert_image_to_bytes(base64_image)
    assert result == b'Test image data'


@patch('requests.get')
def test_convert_url_image(mock_get):
    # Mock a URL image
    url = 'http://example.com/image.jpg'
    mock_get.return_value = Mock(content=b'Test image data')
    result = convert_image_to_bytes(url)
    assert result == b'Test image data'


if __name__ == "__main__":
    unittest.main()

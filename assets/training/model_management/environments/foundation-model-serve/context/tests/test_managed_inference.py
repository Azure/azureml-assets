# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import unittest
import pytest
from constants import SupportedTask
from managed_inference import MIRPayload, process_input_data_for_text_to_image, get_request_data
from transformers import LlamaTokenizerFast, AutoTokenizer
from unittest.mock import patch, MagicMock


class TestMIRPayload(unittest.TestCase):
    def setUp(self):
        self.sample_inputs_text_generation = {
            "text-generation": [
                {
                    "input_data": ["the meaning of life is"],
                    "params": {"max_new_tokens": 256, "do_sample": True},
                },
                {
                    "input_data": [
                        "The recipe of a good movie is",
                        "Quantum physics is",
                        "the meaning of life is",
                    ],
                    "params": {"max_new_tokens": 256, "do_sample": True},
                },
            ]
        }
        self.sample_inputs_chat_completion = {
            "chat-completion": [
                {
                    "input_data": {
                        "input_string": [
                            {
                                "role": "user",
                                "content": "What is the tallest building in the world?",
                            },
                            {
                                "role": "assistant",
                                "content": (
                                    "As of 2021, the Burj Khalifa in Dubai, United Arab Emirates is the "
                                    "tallest building in the world, standing at a height of 828 meters "
                                    "(2,722 feet). It was completed in 2010 and has 163 floors. The Burj "
                                    "Khalifa is not only the tallest building in the world but also holds "
                                    "several other records, such as the highest occupied floor, highest "
                                    "outdoor observation deck, elevator with the longest travel distance, "
                                    "and the tallest freestanding structure in the world."
                                ),
                            },
                            {"role": "user", "content": "and in Africa?"},
                            {
                                "role": "assistant",
                                "content": (
                                    "In Africa, the tallest building is the Carlton Centre, located in "
                                    "Johannesburg, South Africa. It stands at a height of 50 floors and 223 "
                                    "meters (730 feet). The CarltonDefault Centre was completed in 1973 and "
                                    "was the tallest building in Africa for many years until the construction"
                                    " of the Leonardo, a 55-story skyscraper in Sandton, Johannesburg, which "
                                    "was completed in 2019 and stands at a height of 230 meters (755 feet). "
                                    "Other notable tall buildings in Africa include the Ponte City Apartments"
                                    " in Johannesburg, the John Hancock Center in Lagos, Nigeria, and the "
                                    "Alpha II Building in Abidjan, Ivory Coast"
                                ),
                            },
                            {"role": "user", "content": "and in Europe?"},
                        ],
                        "parameters": {
                            "temperature": 0.9,
                            "top_p": 0.6,
                            "do_sample": True,
                            "max_new_tokens": 100,
                        },
                    }
                }
            ]
        }
        self.sample_inputs_chat_completion_multimodal = {
            "chat-completion-multimodal": [
                {
                    "input_data": {
                        "input_string": [
                            {
                                "role": "user", "content": [
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                                            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
                                                                  },
                                    },
                                    {
                                        "type": "image_url", "image_url": {
                                            "url": "https://www.ilankelman.org/stopsigns/australia.jpg"},
                                    },
                                    {
                                        "type": "text", "text": "What are in these images? What is the difference between two images?", }, ], }], "parameters": {
                            "temperature": 0.7, "top_p": 0.6, "do_sample": True, "max_new_tokens": 200, }, }}]}
        self.valid_text_gen_queries = [
            ["the meaning of life is"],
            [
                "The recipe of a good movie is",
                "Quantum physics is",
                "the meaning of life is",
            ],
        ]
        self.valid_text_gen_params = [{"max_new_tokens": 256, "do_sample": True}]
        self.valid_chat_completion_queries = [
            (
                "<s>[INST] What is the tallest building in the world? [/INST] As of "
                "2021, the Burj Khalifa in Dubai, United Arab Emirates is the tallest "
                "building in the world, standing at a height of 828 meters (2,722 "
                "feet). It was completed in 2010 and has 163 floors. The Burj Khalifa "
                "is not only the tallest building in the world but also holds several "
                "other records, such as the highest occupied floor, highest outdoor "
                "observation deck, elevator with the longest travel distance, and the "
                "tallest freestanding structure in the world. </s><s>[INST] and in "
                "Africa? [/INST] In Africa, the tallest building is the Carlton Centre"
                ", located in Johannesburg, South Africa. It stands at a height of 50 "
                "floors and 223 meters (730 feet). The CarltonDefault Centre was "
                "completed in 1973 and was the tallest building in Africa for many "
                "years until the construction of the Leonardo, a 55-story skyscraper "
                "in Sandton, Johannesburg, which was completed in 2019 and stands at "
                "a height of 230 meters (755 feet). Other notable tall buildings in "
                "Africa include the Ponte City Apartments in Johannesburg, the John "
                "Hancock Center in Lagos, Nigeria, and the Alpha II Building in "
                "Abidjan, Ivory Coast </s><s>[INST] and in Europe? [/INST]"
            ),
            (
                "<s>[INST] Describe the contents of the image at URL: http://example.com/image1.jpg. "
                "[/INST] <s>[INST] What can you infer from the description provided? [/INST]"
            ),
        ]
        self.valid_chat_completion_params = [
            {"temperature": 0.9, "top_p": 0.6, "do_sample": True, "max_new_tokens": 100}
        ]
        self.valid_chat_completion_multimodal_queries = [
            ("<s>[INST] Analyze the two images: "
             "Image 1: https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg, "
             "Image 2: https://www.ilankelman.org/stopsigns/australia.jpg. "
             "What are in these images? What is the difference between two images? [/INST]")]

        self.valid_chat_completion_multimodal_params = [
            {"temperature": 0.7, "top_p": 0.6, "do_sample": True, "max_new_tokens": 200}
        ]

    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_from_dict(self, mock_tokenizer):
        llama_tokenizer_fast = LlamaTokenizerFast.from_pretrained(
            "hf-internal-testing/llama-tokenizer", use_default_system_prompt=False
        )
        mock_tokenizer.return_value = llama_tokenizer_fast
        for sample_ip in self.sample_inputs_text_generation["text-generation"]:
            sample_ip.update({"task_type": SupportedTask.TEXT_GENERATION})
            payload = MIRPayload.from_dict(sample_ip)
            self.assertIn(payload.query, self.valid_text_gen_queries)
            self.assertIn(payload.params, self.valid_text_gen_params)

        for sample_ip in self.sample_inputs_chat_completion["chat-completion"]:
            sample_ip.update({"task_type": SupportedTask.CHAT_COMPLETION})
            payload = MIRPayload.from_dict(sample_ip)
            print(payload.query)
            self.assertIn(payload.query, self.valid_chat_completion_queries)
            self.assertIn(payload.params, self.valid_chat_completion_params)

    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_multimodal_payload(self, mock_tokenizer):
        mock_tokenizer_instance = AutoTokenizer.from_pretrained(
            "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
        )
        mock_tokenizer_instance.apply_chat_template = MagicMock(
            return_value="<s>[INST] Analyze the two images: Image 1: https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg, Image 2: https://www.ilankelman.org/stopsigns/australia.jpg. What are in these images? What is the difference between two images? [/INST]")
        mock_tokenizer.return_value = mock_tokenizer_instance

        for sample_ip in self.sample_inputs_chat_completion_multimodal[
            "chat-completion-multimodal"
        ]:
            sample_ip.update({"task_type": SupportedTask.CHAT_COMPLETION})
            payload = MIRPayload.from_dict(sample_ip)

            self.assertIn(payload.query, self.valid_chat_completion_multimodal_queries)
            self.assertIn(payload.params, self.valid_chat_completion_multimodal_params)

    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_convert_query_to_list(self, mock_tokenizer):
        llama_tokenizer_fast = LlamaTokenizerFast.from_pretrained(
            "hf-internal-testing/llama-tokenizer", use_default_system_prompt=False
        )
        mock_tokenizer.return_value = llama_tokenizer_fast
        # Should not convert query parameter to a list of lists
        data = self.sample_inputs_text_generation["text-generation"][0]
        data.update({"task_type": SupportedTask.TEXT_GENERATION})
        payload = MIRPayload.from_dict(data)
        self.assertIsInstance(payload.query, list)
        payload.convert_query_to_list()
        self.assertNotIsInstance(payload.query[0], list)

        # Should convert query parameter to a list
        data = self.sample_inputs_chat_completion["chat-completion"][0]
        data.update({"task_type": SupportedTask.CHAT_COMPLETION})
        payload = MIRPayload.from_dict(data)
        self.assertNotIsInstance(payload.query, list)
        payload.convert_query_to_list()
        self.assertIsInstance(payload.query, list)


def test_process_input_data_for_text_to_image():
    # Create a mock input dictionary
    inputs = {
        "parameters": {"param1": "value1", "param2": "value2"},
        "columns": ["prompt", "negative_prompt"],
        "data": [{"prompt": "prompt sample 1", "negative_prompt": "prompt sample 2"}],
    }

    # Call the function with the mock input
    input_data, params = process_input_data_for_text_to_image(inputs)

    # Check that the function correctly processed the input data
    assert input_data == [
        ["prompt sample 1", "prompt sample 2", None, None]
    ], "The function should correctly process the input data"
    assert params == {
        "param1": "value1",
        "param2": "value2",
    }, "The function should correctly process the parameters"


def test_process_input_data_for_text_to_image_inpainting():
    # Create a mock input dictionary
    inputs = {
        "parameters": {"param1": "value1", "param2": "value2"},
        "columns": ["prompt", "negative_prompt", "mask_image", "image"],
        "data": [
            {
                "prompt": "prompt sample 1",
                "negative_prompt": "prompt sample 2",
                "mask_image": "http://mask",
                "image": "http://image",
            }
        ],
    }

    # Call the function with the mock input
    input_data, params = process_input_data_for_text_to_image(inputs)

    # Check that the function correctly processed the input data
    assert input_data == [
        ["prompt sample 1", "prompt sample 2", "http://image", "http://mask"]
    ], "The function should correctly process the input data"
    assert params == {
        "param1": "value1",
        "param2": "value2",
    }, "The function should correctly process the parameters"


def test_process_input_data_for_text_to_image_exception():
    # Create a mock input dictionary with missing keys
    inputs = {
        "parameters": {"param1": "value1", "param2": "value2"},
        "data": ["prompt1"],
    }

    # Call the function with the mock input and check that it raises an exception
    with pytest.raises(Exception):
        process_input_data_for_text_to_image(inputs)


def test_get_request_data_text_gen_preview_format():
    # Create a mock request data
    data = {
        "input_data": {
            "input_string": ["prompt sample 1", "prompt sample 2"],
            "parameters": {"param1": "value1", "param2": "value2"},
        },
        "task_type": SupportedTask.TEXT_GENERATION,
    }

    # Call the function with the mock data
    input_data, params, task_type, is_preview_format = get_request_data(data)

    # Check that the function correctly processed the request data
    assert input_data == ["prompt sample 1", "prompt sample 2"]
    assert params == {"param1": "value1", "param2": "value2"}
    assert task_type == SupportedTask.TEXT_GENERATION
    assert is_preview_format is True


def test_get_request_data_text_gen_new_format():
    # Create a mock request data
    data = {
        "input_data": ["prompt sample 1", "prompt sample 2"],
        "params": {"param1": "value1", "param2": "value2"},
        "task_type": SupportedTask.TEXT_GENERATION,
    }

    # Call the function with the mock data
    input_data, params, task_type, is_preview_format = get_request_data(data)

    # Check that the function correctly processed the request data
    assert input_data == ["prompt sample 1", "prompt sample 2"]
    assert params == {"param1": "value1", "param2": "value2"}
    assert task_type == SupportedTask.TEXT_GENERATION
    assert is_preview_format is False


def test_get_request_data_text_to_image():
    # Create a mock request data
    data = {
        "input_data": {
            "parameters": {"param1": "value1", "param2": "value2"},
            "columns": ["prompt", "negative_prompt"],
            "data": [
                {"prompt": "prompt sample 1", "negative_prompt": "prompt sample 2"}
            ],
        },
        "task_type": SupportedTask.TEXT_TO_IMAGE,
    }

    # Call the function with the mock data
    input_data, params, task_type, is_preview_format = get_request_data(data)

    # Check that the function correctly processed the request data
    assert input_data == [["prompt sample 1", "prompt sample 2", None, None]]
    assert params == {"param1": "value1", "param2": "value2"}
    assert task_type == SupportedTask.TEXT_TO_IMAGE
    assert is_preview_format is True


def test_get_request_data_text_to_image_inpainting():
    # Create a mock request data
    data = {
        "input_data": {
            "parameters": {"param1": "value1", "param2": "value2"},
            "columns": ["prompt", "negative_prompt", "image", "mask_image"],
            "data": [
                {
                    "prompt": "prompt sample 1",
                    "negative_prompt": "prompt sample 2",
                    "image": "http://image",
                    "mask_image": "http://mask",
                }
            ],
        },
        "task_type": SupportedTask.TEXT_TO_IMAGE,
    }

    # Call the function with the mock data
    input_data, params, task_type, is_preview_format = get_request_data(data)

    # Check that the function correctly processed the request data
    assert input_data == [
        ["prompt sample 1", "prompt sample 2", "http://image", "http://mask"]
    ]
    assert params == {"param1": "value1", "param2": "value2"}
    assert task_type == SupportedTask.TEXT_TO_IMAGE
    assert is_preview_format is True


if __name__ == "__main__":
    unittest.main()





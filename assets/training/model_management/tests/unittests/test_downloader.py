# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Test download."""

import unittest
from unittest.mock import patch
from pathlib import Path
from azureml.model.mgmt.downloader.downloader import (
    HuggingfaceDownloader,
    GITDownloader,
    AzureBlobstoreDownloader,
    download_model,
)
from azureml.model.mgmt.utils import common_utils
from azureml.model.mgmt.utils.common_utils import create_namespace_from_dict
from azureml.model.mgmt.config import LlamaHFModels, LlamaModels


class TestDownloaders(unittest.TestCase):
    """Test downloaders."""

    def test_huggingface_downloader(self):
        """Test huggingface downloader."""
        model_id = "username/model_name"
        download_dir = Path("path/to/download/dir")
        token = None

        with patch("huggingface_hub.hf_api.HfApi") as MockHfApi, patch(
            "azureml.model.mgmt.downloader.downloader.HuggingfaceDownloader._download"
        ) as MockGITDownloader:
            mock_api = MockHfApi.return_value
            mock_api.list_models.return_value = [
                create_namespace_from_dict(
                    {
                        "modelId": "username/model_name",
                        "pipeline_tag": "task_name",
                        "tags": ["tag1", "tag2", "tag3"],
                        "sha": "123456789",
                    }
                )
            ]

            common_utils.hf_api = mock_api
            downloader = HuggingfaceDownloader(model_id, download_dir, token)
            downloader.download_model()

            self.assertEqual(downloader.model_info.modelId, model_id)
            self.assertEqual(downloader.model_info.pipeline_tag, "task_name")
            self.assertEqual(downloader.model_info.sha, "123456789")
            self.assertEqual(mock_api.list_models.call_count, 1)
            self.assertEqual(MockGITDownloader.call_count, 1)

    def test_git_downloader(self):
        """Test git downloader."""
        model_uri = "https://github.com/some_model"
        download_dir = Path("path/to/download/dir")

        with patch("azureml.model.mgmt.downloader.downloader.GITDownloader._download"):
            downloader = GITDownloader(model_uri, download_dir)
            download_details = downloader.download_model()

            self.assertEqual(download_details["name"], "some_model")
            self.assertEqual(download_details["tags"], {})
            self.assertEqual(download_details["properties"], {})

    def test_azure_blobstore_downloader(self):
        """Test blobstorage downloader."""
        model_uri = "https://blobstorageaccount.blob.core.windows.net/models/model_folder"
        download_dir = Path("path/to/download/dir")

        with patch("azureml.model.mgmt.downloader.downloader.AzureBlobstoreDownloader._download"):
            downloader = AzureBlobstoreDownloader(model_uri, download_dir)
            download_details = downloader.download_model()
            self.assertEqual(download_details["name"], "model_folder")
            self.assertEqual(download_details["tags"], {})
            self.assertEqual(download_details["properties"], {})


class TestDownloadModel(unittest.TestCase):
    """Test model download."""

    def test_download_model_with_huggingface_source(self):
        """Test huggingface model download."""
        model_source = "Huggingface"
        model_id = "test_model_id"
        download_dir = Path("path/to/download/dir")
        token = None

        with patch("azureml.model.mgmt.downloader.downloader.HuggingfaceDownloader") as MockDownloader:
            mock_downloader = MockDownloader.return_value
            mock_downloader.download_model.return_value = {
                "name": "model_name",
                "tags": {"tag1": "value1"},
                "properties": {"prop1": "value1"},
            }

            result = download_model(model_source, model_id, download_dir, token)

            self.assertEqual(
                result,
                {
                    "name": "model_name",
                    "tags": {"tag1": "value1"},
                    "properties": {"prop1": "value1"},
                },
            )
            mock_downloader.download_model.assert_called_once()

    def test_download_model_with_git_source(self):
        """Test GIT model download."""
        model_source = "GIT"
        model_id = "https://github.com/some_model"
        download_dir = Path("path/to/download/dir")
        token = None

        with patch("azureml.model.mgmt.downloader.downloader.GITDownloader") as MockDownloader:
            mock_downloader = MockDownloader.return_value
            mock_downloader.download_model.return_value = {
                "name": "some_model",
                "tags": {"tag2": "value2"},
                "properties": {"prop2": "value2"},
            }

            result = download_model(model_source, model_id, download_dir, token)

            self.assertEqual(
                result,
                {
                    "name": "some_model",
                    "tags": {"tag2": "value2"},
                    "properties": {"prop2": "value2"},
                },
            )
            mock_downloader.download_model.assert_called_once()

    def test_download_model_with_azureblob_source(self):
        """Test blobstorage model download."""
        model_source = "AzureBlob"
        model_id = "https://blobstorageaccount.blob.core.windows.net/models/model_folder"
        download_dir = Path("path/to/download/dir")
        token = None

        with patch("azureml.model.mgmt.downloader.downloader.AzureBlobstoreDownloader") as MockDownloader:
            mock_downloader = MockDownloader.return_value
            mock_downloader.download_model.return_value = {
                "name": "model_folder",
                "tags": {"tag3": "value3"},
                "properties": {"prop3": "value3"},
            }

            result = download_model(model_source, model_id, download_dir, token)

            self.assertEqual(
                result,
                {
                    "name": "model_folder",
                    "tags": {"tag3": "value3"},
                    "properties": {"prop3": "value3"},
                },
            )
            mock_downloader.download_model.assert_called_once()

    def test_fixed_llama_models_list(self):
        """Test llama model list."""
        list_of_models = LlamaModels.list_values()

        allowed_llama_models = ["meta-llama/Llama-2-7b-chat",
                                "meta-llama/Llama-2-13b-chat",
                                "meta-llama/Llama-2-70b-chat",
                                "meta-llama/Llama-2-7b",
                                "meta-llama/Llama-2-13b",
                                "meta-llama/Llama-2-70b"
                                ]
        for model in list_of_models:
            self.assertIn(model, allowed_llama_models)

    def test_fixed_llama_hf_models_list(self):
        """Test llama hf model list."""
        list_of_models = LlamaHFModels.list_values()

        allowed_llama_hf_models = ["meta-llama/Llama-2-7b-chat-hf",
                                   "meta-llama/Llama-2-13b-chat-hf",
                                   "meta-llama/Llama-2-70b-chat-hf",
                                   "meta-llama/Llama-2-7b-hf",
                                   "meta-llama/Llama-2-13b-hf",
                                   "meta-llama/Llama-2-70b-hf"
                                   ]
        for model in list_of_models:
            self.assertIn(model, allowed_llama_hf_models)

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
from azureml.model.mgmt.utils.common_utils import create_namespace_from_dict


class TestDownloaders(unittest.TestCase):
    """Test downloaders."""

    def test_huggingface_downloader(self):
        model_id = "username/model_name"
        download_dir = Path("path/to/download/dir")

        with patch("huggingface_hub.hf_api.HfApi") as MockHfApi, \
                patch("azureml.model.mgmt.downloader.download_utils._download_git_model") as MockGITDownloader:
            mock_api = MockHfApi.return_value
            mock_api.list_models.return_value = [
                create_namespace_from_dict({
                    "modelId": "username/model_name",
                    "pipeline_tag": "task_name",
                    "tags": ["tag1", "tag2", "tag3"],
                    "sha": "123456789",
                })
            ]

            downloader = HuggingfaceDownloader(model_id)
            downloader._hf_api = mock_api
            downloader.download_model(download_dir)

            self.assertEqual(downloader.model_info.modelId, model_id)
            self.assertEqual(downloader.model_info.pipeline_tag, "task_name")
            self.assertEqual(downloader.model_info.sha, "123456789")
            self.assertEqual(mock_api.list_models.call_count, 1)
            self.assertEqual(MockGITDownloader.call_count, 1)

    def test_git_downloader(self):
        model_uri = "https://github.com/some_model"
        download_dir = Path("path/to/download/dir")

        with patch("azureml.model.mgmt.downloader.download_utils._download_git_model"):
            downloader = GITDownloader(model_uri)
            download_details = downloader.download_model(download_dir)

            self.assertEqual(download_details["name"], "some_model")
            self.assertEqual(download_details["tags"], {})
            self.assertEqual(download_details["properties"], {})

    def test_azure_blobstore_downloader(self):
        model_uri = "https://blobstorageaccount.blob.core.windows.net/models/model_folder"
        download_dir = Path("path/to/download/dir")

        with patch("azureml.model.mgmt.downloader.download_utils._download_azure_artifacts"):
            downloader = AzureBlobstoreDownloader(model_uri)
            download_details = downloader.download_model(download_dir)
            self.assertEqual(download_details["name"], "model_folder")
            self.assertEqual(download_details["tags"], {})
            self.assertEqual(download_details["properties"], {})


class TestDownloadModel(unittest.TestCase):
    """Test model download."""

    def test_download_model_with_huggingface_source(self):
        model_source = "Huggingface"
        model_id = "test_model_id"
        download_dir = Path("path/to/download/dir")

        with patch("azureml.model.mgmt.downloader.downloader.HuggingfaceDownloader") as MockDownloader:
            mock_downloader = MockDownloader.return_value
            mock_downloader.download_model.return_value = {
                "name": "model_name",
                "tags": {"tag1": "value1"},
                "properties": {"prop1": "value1"},
            }

            result = download_model(model_source, model_id, download_dir)

            self.assertEqual(result, {
                "name": "model_name",
                "tags": {"tag1": "value1"},
                "properties": {"prop1": "value1"},
            })
            mock_downloader.download_model.assert_called_once_with(download_dir)

    def test_download_model_with_git_source(self):
        model_source = "GIT"
        model_id = "https://github.com/some_model"
        download_dir = Path("path/to/download/dir")

        with patch("azureml.model.mgmt.downloader.downloader.GITDownloader") as MockDownloader:
            mock_downloader = MockDownloader.return_value
            mock_downloader.download_model.return_value = {
                "name": "some_model",
                "tags": {"tag2": "value2"},
                "properties": {"prop2": "value2"},
            }

            result = download_model(model_source, model_id, download_dir)

            self.assertEqual(result, {
                "name": "some_model",
                "tags": {"tag2": "value2"},
                "properties": {"prop2": "value2"},
            })
            mock_downloader.download_model.assert_called_once_with(download_dir)

    def test_download_model_with_azureblob_source(self):
        model_source = "AzureBlob"
        model_id = "https://blobstorageaccount.blob.core.windows.net/models/model_folder"
        download_dir = Path("path/to/download/dir")

        with patch("azureml.model.mgmt.downloader.downloader.AzureBlobstoreDownloader") as MockDownloader:
            mock_downloader = MockDownloader.return_value
            mock_downloader.download_model.return_value = {
                "name": "model_folder",
                "tags": {"tag3": "value3"},
                "properties": {"prop3": "value3"},
            }

            result = download_model(model_source, model_id, download_dir)

            self.assertEqual(result, {
                "name": "model_folder",
                "tags": {"tag3": "value3"},
                "properties": {"prop3": "value3"},
            })
            mock_downloader.download_model.assert_called_once_with(download_dir)

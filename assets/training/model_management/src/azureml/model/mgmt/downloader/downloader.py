# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Downloader Module."""

import langcodes
from azureml._common._error_definition import AzureMLError
from azureml._common.exceptions import AzureMLException
from azureml.model.mgmt.config import PathType
from azureml.model.mgmt.downloader.config import ModelSource
from azureml.model.mgmt.downloader.download_utils import download_model_for_path_type
from huggingface_hub.hf_api import ModelInfo
from pathlib import Path
from azureml.model.mgmt.utils.common_utils import retry, fetch_huggingface_model_info
from azureml.model.mgmt.utils.exceptions import InvalidHuggingfaceModelIDError
from azureml.model.mgmt.utils.logging_utils import get_logger


logger = get_logger(__name__)


PROPERTIES = [
    "SHA",
    "last_modified",
    "model_id",
    "size",
    "datasets",
    "languages",
    "finetuning_tasks",
]
TAGS = ["task", "license"]


class HuggingfaceDownloader:
    """Huggingface model downloader class."""

    HF_ENDPOINT = "https://huggingface.co"
    URI_TYPE = PathType.GIT.value

    # Valid language codes which conflicts with other model tags.
    # jax for example is a NumPy framework and is present in tags for most HF models.
    # jax is also a language code used to represent the language spoken by the Jaintia people.
    LANGUAGE_CODE_EXCEPTIONS = ["jax", "vit"]

    def __init__(self, model_id: str):
        """Huggingface downloader init.

        param model_id: https://huggingface.co/<model_id>
        type model_id: str
        """
        self._model_id = model_id
        self._model_uri = self.HF_ENDPOINT + f"/{model_id}"
        self._model_info = None

    @property
    def model_info(self) -> ModelInfo:
        """Hugging face model info."""
        if self._model_info is None:
            self._model_info = fetch_huggingface_model_info(self._model_id)
        return self._model_info

    def _get_model_properties(self):
        languages = []
        datasets = []
        misc = []
        all_tags = self.model_info.tags
        props = {
            "model_id": self.model_info.modelId,
            "task": self.model_info.pipeline_tag,
            "SHA": self.model_info.sha,
        }

        for tag in all_tags:
            if langcodes.tag_is_valid(tag) and tag not in HuggingfaceDownloader.LANGUAGE_CODE_EXCEPTIONS:
                languages.append(tag)
            elif tag.startswith("dataset:"):
                datasets.append(tag.split(":")[1])
            elif tag.startswith("license:"):
                props["license"] = tag.split(":")[1]
            else:
                misc.append(tag.lower())

        if datasets:
            props["datasets"] = ", ".join(datasets)
        if languages:
            props["languages"] = ", ".join(languages)
        if misc:
            props["misc"] = misc

        return props

    def download_model(self, download_dir):
        """Download a Hugging face model and return details."""
        if self.model_info:
            download_details = download_model_for_path_type(self.URI_TYPE, self._model_uri, download_dir)
            model_props = self._get_model_properties()
            model_props.update(download_details)
            tags = {k: model_props[k] for k in TAGS if k in model_props}
            props = {k: model_props[k] for k in PROPERTIES if k in model_props}
            return {
                "name": "-".join(self._model_id.split("/")),
                "tags": tags,
                "properties": props,
                "misc": model_props.get("misc"),
            }
        else:
            raise AzureMLException._with_error(
                AzureMLError.create(InvalidHuggingfaceModelIDError, model_id=self._model_id)
            )


class GITDownloader:
    """Downloader class for model hosted on public git repositories."""

    URI_TYPE = PathType.GIT.value

    def __init__(self, model_uri):
        """GIT downloader init.

        param model_uri: GIT repository URL for the model.
            eg: https://github.com/some_model
        type model_uri: str
        """
        self._model_uri = model_uri

    @retry(3)
    def download_model(self, download_dir):
        """Download a publicly hosted GIT model and return details."""
        download_details = download_model_for_path_type(self.URI_TYPE, self._model_uri, download_dir)
        tags = {k: download_details[k] for k in TAGS if k in download_details}
        props = {k: download_details[k] for k in PROPERTIES if k in download_details}
        return {
            "name": self._model_uri.split("/")[-1],
            "tags": tags,
            "properties": props,
        }


class AzureBlobstoreDownloader:
    """Downloader class for model hosted on a public azure blobstorage."""

    URI_TYPE = PathType.AZUREBLOB.value

    def __init__(self, model_uri: str):
        """Azure blobstore downloader init.

        param model_uri: blobstore path to the model.
            eg: https://blobstorageaccount.blob.core.windows.net/models/model_folder
        type model_uri: str
        """
        self._model_uri = model_uri

    @retry(3)
    def download_model(self, download_dir):
        """Download a model from a publicly accessible azure blobstorage and return details."""
        download_details = download_model_for_path_type(self.URI_TYPE, self._model_uri, download_dir)
        tags = {k: download_details[k] for k in TAGS if k in download_details}
        props = {k: download_details[k] for k in PROPERTIES if k in download_details}
        return {
            "name": self._model_uri.split("/")[-1],
            "tags": tags,
            "properties": props,
        }


def download_model(model_source: str, model_id: str, download_dir: Path):
    """Download model and return model information."""
    if model_source == ModelSource.HUGGING_FACE.value:
        downloader = HuggingfaceDownloader(model_id=model_id)
    elif model_source == ModelSource.GIT.value:
        downloader = GITDownloader(model_uri=model_id)
    elif model_source == ModelSource.AZUREBLOB.value:
        downloader = AzureBlobstoreDownloader(model_uri=model_id)
    else:
        raise Exception(f"Download from {model_source} is not supported")
    return downloader.download_model(download_dir)

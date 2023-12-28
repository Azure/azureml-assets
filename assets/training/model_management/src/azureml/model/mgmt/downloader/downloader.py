# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Downloader Module."""

import langcodes
import os
import shutil
import stat
from azureml._common._error_definition import AzureMLError
from azureml._common.exceptions import AzureMLException
from azureml.model.mgmt.config import PathType
from azureml.model.mgmt.downloader.config import ModelSource
from azureml.model.mgmt.utils.common_utils import (
    BUFFER_SPACE,
    run_command,
    log_execution_time,
    round_size,
    get_system_time_utc,
    get_file_or_folder_size,
    get_filesystem_available_space_in_kb,
    get_git_lfs_blob_size_in_kb,
)
from azureml.model.mgmt.utils.exceptions import BlobStorageDownloadError, GITCloneError, VMNotSufficientForOperation
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


class AzureBlobstoreDownloader:
    """Downloader class for model hosted on a public azure blobstorage."""

    URI_TYPE = PathType.AZUREBLOB.value

    def __init__(self, model_uri: str, download_dir: Path):
        """Azure blobstore downloader init.

        param model_uri: blobstore path to the model.
            eg: https://blobstorageaccount.blob.core.windows.net/models/model_folder
        type model_uri: str
        param download_dir: Directory path to download artifacts to
        type download_dir: Path
        """
        self._model_uri = model_uri
        self._download_dir = download_dir

    @log_execution_time
    def _download(self):
        try:
            logger.info(f"self._model_uri: {self._model_uri}")
            logger.info(f"self._download_dir: {self._download_dir}")
            # Ensure trailing slash in the download directory
            download_dir = str(self._download_dir)
            if not download_dir.endswith("/"):
                download_dir += "/"
            # Remove trailing slash from the model URI
            model_uri = self._model_uri.rstrip('/')
            
            # Extract the model name from the URI
            model_name = model_uri.split("/")[-1]
            logger.info(f"model_name: {model_name}")

            # Construct the target download directory with the model name
            target_download_dir = os.path.join(download_dir, model_name)
            logger.info(f"target_download_dir: {target_download_dir}")

            download_cmd = f"azcopy cp --recursive=true '{self._model_uri}' {target_download_dir}"
            # TODO: Handle error case correctly, since azcopy exits with 0 exit code, even in case of error.
            # https://github.com/Azure/azureml-assets/issues/283
            exit_code, stdout = run_command(download_cmd)
            if exit_code != 0:
                raise AzureMLException._with_error(
                    AzureMLError.create(BlobStorageDownloadError, uri=self._model_uri, error=stdout)
                )
            # Log the contents of the downloaded directory
            downloaded_files = list(Path(target_download_dir).rglob("*"))
            logger.info(f"Contents of the downloaded directory: {downloaded_files}")
            
            return {
                "download_time_utc": get_system_time_utc(),
                "size": round_size(get_file_or_folder_size(self._download_dir)),
            }
        except Exception as e:
            raise AzureMLException._with_error(
                AzureMLError.create(BlobStorageDownloadError, uri=self._model_uri, error=e)
            )

    @retry(3)
    def download_model(self):
        """Download a model from a publicly accessible azure blobstorage and return details."""
        download_details = self._download()
        tags = {k: download_details[k] for k in TAGS if k in download_details}
        props = {k: download_details[k] for k in PROPERTIES if k in download_details}
        return {
            "name": self._model_uri.split("/")[-1],
            "tags": tags,
            "properties": props,
        }


class GITDownloader:
    """Downloader class for model hosted on public git repositories."""

    URI_TYPE = PathType.GIT.value
    FILES_TO_REMOVE = [".gitattributes"]
    DIRS_TO_REMOVE = [".git"]

    def __init__(self, model_uri: str, download_dir: Path):
        """GIT downloader init.

        param model_uri: GIT repository URL for the model.
            eg: https://github.com/some_model
        type model_uri: str
        param download_dir: Directory path to download artifacts to
        type download_dir: Path
        """
        self._model_uri = model_uri
        self._download_dir = download_dir

    def _onerror(func, path, exc_info):
        """Error Handler for shutil rmtree."""
        if not os.access(path, os.W_OK):
            os.chmod(path, stat.S_IWUSR)
            func(path)
        else:
            raise

    @log_execution_time
    def _download(self):
        try:
            # do shallow fetch
            logger.info("Cloning non LFS files first")
            clone_cmd = f"GIT_LFS_SKIP_SMUDGE=1 git clone --depth=1 {self._model_uri} {self._download_dir}"
            exit_code, stdout = run_command(clone_cmd)
            if exit_code != 0:
                raise AzureMLException._with_error(
                    AzureMLError.create(GITCloneError, uri=self._model_uri, error=stdout)
                )

            logger.info("Done. Checking if its safe to pull LFS files")
            available_size_on_vm = get_filesystem_available_space_in_kb()
            total_blob_size_in_kb = get_git_lfs_blob_size_in_kb(git_dir=self._download_dir)
            if available_size_on_vm - total_blob_size_in_kb < BUFFER_SPACE:
                details = (
                    f"Compute not sufficient to download model files keeping 1GB buffer space. "
                    f"AVAILABLE_SIZE_IN_KB = {available_size_on_vm} "
                    f"Remaining model size to download: {total_blob_size_in_kb}. "
                    "Please select optimal SKU for your model size from here: "
                    "https://learn.microsoft.com/en-us/azure/virtual-machines/sizes"
                )
                raise AzureMLException._with_error(
                    AzureMLError.create(VMNotSufficientForOperation, operation="download", details=details)
                )

            logger.info("Done. Downloading LFS files")
            clone_cmd = f"cd {self._download_dir} && git lfs pull"
            exit_code, stdout = run_command(clone_cmd)
            if exit_code != 0:
                raise AzureMLException._with_error(
                    AzureMLError.create(GITCloneError, uri=self._model_uri, error=stdout)
                )

            for dir_path in GITDownloader.DIRS_TO_REMOVE:
                path = os.path.join(self._download_dir, dir_path)
                if os.path.exists(path) and os.path.isdir(path):
                    shutil.rmtree(path, onerror=self._onerror)

            for file_path in GITDownloader.FILES_TO_REMOVE:
                path = os.path.join(self._download_dir, file_path)
                if os.path.exists(path) and os.path.isfile(path):
                    os.remove(path)

            return {
                "download_time_utc": get_system_time_utc(),
                "size": round_size(get_file_or_folder_size(self._download_dir)),
            }
        except Exception as e:
            raise AzureMLException._with_error(AzureMLError.create(GITCloneError, uri=self._model_uri, error=e))

    @retry(3)
    def download_model(self):
        """Download a publicly hosted GIT model and return details."""
        download_details = self._download()
        tags = {k: download_details[k] for k in TAGS if k in download_details}
        props = {k: download_details[k] for k in PROPERTIES if k in download_details}
        return {
            "name": self._model_uri.split("/")[-1],
            "tags": tags,
            "properties": props,
        }


class HuggingfaceDownloader(GITDownloader):
    """Huggingface model downloader class."""

    HF_ENDPOINT = "https://huggingface.co"
    URI_TYPE = PathType.GIT.value

    # Valid language codes which conflicts with other model tags.
    # jax for example is a NumPy framework and is present in tags for most HF models.
    # jax is also a language code used to represent the language spoken by the Jaintia people.
    LANGUAGE_CODE_EXCEPTIONS = ["jax", "vit"]

    def __init__(self, model_id: str, download_dir: Path):
        """Huggingface downloader init.

        param model_id: https://huggingface.co/<model_id>
        type model_id: str
        param download_dir: Directory path to download artifacts to
        type download_dir: Path
        """
        self._model_id = model_id
        self._model_uri = self.HF_ENDPOINT + f"/{model_id}"
        self._download_dir = download_dir
        self._model_info = None
        super().__init__(self._model_uri, download_dir)

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

    def download_model(self):
        """Download a Hugging face model and return details."""
        if self.model_info:
            download_details = self._download()
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


def download_model(model_source: str, model_id: str, download_dir: Path):
    """Download model and return model information."""
    if model_source == ModelSource.HUGGING_FACE.value:
        downloader = HuggingfaceDownloader(model_id=model_id, download_dir=download_dir)
    elif model_source == ModelSource.GIT.value:
        downloader = GITDownloader(model_uri=model_id, download_dir=download_dir)
    elif model_source == ModelSource.AZUREBLOB.value:
        downloader = AzureBlobstoreDownloader(model_uri=model_id, download_dir=download_dir)
    else:
        raise Exception(f"Download from {model_source} is not supported")
    return downloader.download_model()

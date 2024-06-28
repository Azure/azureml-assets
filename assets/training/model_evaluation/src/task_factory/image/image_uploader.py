# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Image uploader (store images in datastore and pass URLs instead of image data)."""
# TODO: refactor to use the same image uploading code in the Dataset Downloader and the Compute Metrics components.

import base64
import io
import os
import tempfile
import uuid

from contextlib import AbstractContextManager
from typing import List, Tuple, Union

from PIL import Image

from azureml.core import Run, Datastore


_DATASTORE_DIRECTORY_URL_TEMPLATE = "AmlDatastore://{datastore_name}/{directory_name}"
_RANDOM_IMAGE_DIRECTORY_TEMPLATE = "images/{random_id}"
_IMAGE_FILE_NAME_TEMPLATE = "image_{image_counter:09d}.png"
_IMAGE_URL_TEMPLATE = "{datastore_directory_url}/{image_file_name}"

_WRONG_IMAGE_TYPE = "Can only upload images passed as PIL.Image's and b64 encoded strings."


def _get_default_datastore() -> Datastore:
    """Get the default datastore for the current run."""
    run = Run.get_context()
    workspace = run.experiment.workspace
    datastore = workspace.get_default_datastore()
    return datastore


def _get_datastore_image_directory_name(datastore_name: str) -> Tuple[str, str]:
    """Make random name for image directory on datastore and get its URL."""
    datastore_directory_name = _RANDOM_IMAGE_DIRECTORY_TEMPLATE.format(random_id=str(uuid.uuid4()))
    datastore_directory_url = _DATASTORE_DIRECTORY_URL_TEMPLATE.format(
        datastore_name=datastore_name, directory_name=datastore_directory_name
    )
    return datastore_directory_name, datastore_directory_url


def _get_image_file_name(image_counter: int) -> str:
    """Make name for local image file."""
    return _IMAGE_FILE_NAME_TEMPLATE.format(image_counter=image_counter)


def _get_image_url(datastore_directory_url: str, image_file_name: str) -> str:
    """Make the image url given the local image file."""
    return _IMAGE_URL_TEMPLATE.format(
        datastore_directory_url=datastore_directory_url, image_file_name=image_file_name
    )


class ImageUploader(AbstractContextManager):
    """Uploader for image data.

    Used to avoid passing image data directly between components (image URLs are passed instead). Passing URLs is
    preferable because it accommodates more data sources, transfers much smaller amounts of data and is easier to
    debug.
    """

    def __init__(self):
        """Initialize upload related members to invalid state."""
        # Set the datastore and local directory related members to `None`.
        self.datastore, self.datastore_directory_name, self.datastore_directory_url = None, None, None
        self.temporary_directory = None

        # Set the image counter and image URL list to `None`.
        self.image_counter = None
        self.image_urls = None

    def __enter__(self):
        """Prepare datastore info and temporary local folder."""
        # Get the default datastore and the directory within it where the images will be uploaded.
        self.datastore = _get_default_datastore()
        self.datastore_directory_name, self.datastore_directory_url = _get_datastore_image_directory_name(
            self.datastore.name
        )

        # Make the local temporary directory.
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.temporary_directory.__enter__()

        # Initialize the image counter and image URL list.
        self.image_counter = 0
        self.image_urls = []

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Upload from temporary local folder to datastore."""
        # Upload the images to datastore.
        self.datastore.upload(src_dir=self.temporary_directory.name, target_path=self.datastore_directory_name)

        # Delete the local temporary directory.
        self.temporary_directory.__exit__(exc_type, exc_value, traceback)

    def upload(self, image: Union[Image.Image, str]) -> None:
        """Save image to temporary local folder.

        :param image: Image, either in PIL format or as base 64 encoded bytes.
        :type image: `PIL.Image` or `str`
        :return: None.
        :rtype: NoneType
        """
        # Convert image data to `PIL.Image`, raising exception if not possible.
        if not isinstance(image, (Image.Image, str)):
            raise ValueError(_WRONG_IMAGE_TYPE)
        if isinstance(image, str):
            image = Image.open(io.BytesIO(base64.b64decode(image)))

        # Save the current image to the local temporary directory.
        image_file_name = _get_image_file_name(image_counter=self.image_counter)
        image_file_path = os.path.join(self.temporary_directory.name, image_file_name)
        image.save(image_file_path)

        # Make the image url and accumulate it.
        image_url = _get_image_url(self.datastore_directory_url, image_file_name)
        self.image_urls.append(image_url)

        # Move to the next image.
        self.image_counter += 1

    @property
    def urls(self) -> List[str]:
        """Get the URLs of the uploaded images.

        :return: Image URLs.
        :rtype: List[str]
        """
        return self.image_urls

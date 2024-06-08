"""Context manager based image uploader."""

# TODO: refactor to use the same image uploading code in the Dataset Downloader and in Compute Metrics components.

import base64
import io
import os
import tempfile
import uuid

from contextlib import AbstractContextManager
from typing import List, Tuple, Union

from PIL import Image

from azureml.core import Run, Datastore


DATASTORE_DIRECTORY_URL_TEMPLATE = "AmlDatastore://{datastore_name}/{directory_name}"
RANDOM_IMAGE_DIRECTORY_TEMPLATE = "images/{random_id}"
IMAGE_FILE_NAME_TEMPLATE = "image_{image_counter:09d}.png"
IMAGE_URL_TEMPLATE = "{datastore_directory_url}/{image_file_name}"


def _get_default_datastore() -> Datastore:
    """Get the default datastore for the current run."""
    run = Run.get_context()
    workspace = run.experiment.workspace
    datastore = workspace.get_default_datastore()
    return datastore


def _get_datastore_image_directory_name(datastore_name: str) -> Tuple[str, str]:
    """Make random name for image directory on datastore and get its URL."""
    datastore_directory_name = RANDOM_IMAGE_DIRECTORY_TEMPLATE.format(random_id=str(uuid.uuid4()))
    datastore_directory_url = DATASTORE_DIRECTORY_URL_TEMPLATE.format(
        datastore_name=datastore_name, directory_name=datastore_directory_name
    )
    return datastore_directory_name, datastore_directory_url


def _get_image_file_name(image_counter: int) -> str:
    """Make name for local image file."""
    return IMAGE_FILE_NAME_TEMPLATE.format(image_counter=image_counter)


def _get_image_url(datastore_directory_url: str, image_file_name: str) -> str:
    """Make the image url given the local image file."""
    return IMAGE_URL_TEMPLATE.format(
        datastore_directory_url=datastore_directory_url, image_file_name=image_file_name
    )


class ImageUploader(AbstractContextManager):
    def __init__(self):
        self.datastore, self.datastore_directory_name, self.datastore_directory_url = None, None, None
        self.image_counter = None

        self.temporary_directory = None

        self.image_urls = []

    def __enter__(self):
        # Get the default datastore and the directory within it where the images will be uploaded.
        self.datastore = _get_default_datastore()
        self.datastore_directory_name, self.datastore_directory_url = _get_datastore_image_directory_name(
            self.datastore.name
        )

        # Make the local temporary directory.
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.temporary_directory.__enter__()

        # Initialize the image counter.
        self.image_counter = 0

    def __exit__(self, *exc_details):
        # Upload the images to datastore.
        self.datastore.upload(src_dir=self.temporary_directory.name, target_path=self.datastore_directory_name)

        # Delete the local temporary directory.
        self.temporary_directory.__exit__(*exc_details)

    def upload_image(self, image: Union[Image.Image, str]) -> None:
        if not isinstance(image, [Image.Image, str]):
            raise

        if isinstance(image, str):
            image = Image.open(io.BytesIO(base64.b64decode(image)))

        # Save the current image to the local temporary directory.
        image_file_name = _get_image_file_name(image_counter=self.image_counter)
        image_file_path = os.path.join(self.temporary_directory.name, image_file_name)
        image.save(image_file_path)

        # Make and accumulate the image url.
        image_url = _get_image_url(self.datastore_directory_url, image_file_name)
        self.image_urls.append(image_url)

    @property
    def urls(self) -> List[str]:
        # Get the urls of the uploaded images.
        return self.image_urls

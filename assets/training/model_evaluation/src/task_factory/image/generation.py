# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""???"""
import base64
import io
import os
import tempfile
import uuid


from typing import Tuple

from PIL import Image

from azureml.core import Run, Datastore
from logging_utilities import get_logger
from task_factory.base import PredictWrapper


# TODO: refactor code to save images to datastore to be used both in Dataset Downloader and in Compute Metrics.
DATASTORE_DIRECTORY_URL_TEMPLATE = "AmlDatastore://{datastore_name}/{directory_name}"
RANDOM_IMAGE_DIRECTORY_TEMPLATE = "images/{random_id}"
IMAGE_FILE_NAME_TEMPLATE = "image_{image_counter:09d}.png"
IMAGE_URL_TEMPLATE = "{datastore_directory_url}/{image_file_name}"

logger = get_logger(name=__name__)


def get_default_datastore() -> Datastore:
    """Get the default datastore for the current run."""
    run = Run.get_context()
    workspace = run.experiment.workspace
    datastore = workspace.get_default_datastore()
    return datastore


def get_datastore_image_directory_name(datastore_name: str) -> Tuple[str, str]:
    """Make random name for image directory on datastore and get its URL."""
    datastore_directory_name = RANDOM_IMAGE_DIRECTORY_TEMPLATE.format(random_id=str(uuid.uuid4()))
    datastore_directory_url = DATASTORE_DIRECTORY_URL_TEMPLATE.format(
        datastore_name=datastore_name, directory_name=datastore_directory_name
    )
    return datastore_directory_name, datastore_directory_url


def get_image_file_name(image_counter: int) -> str:
    """Make name for local image file."""
    return IMAGE_FILE_NAME_TEMPLATE.format(image_counter=image_counter)


def get_image_url(datastore_directory_url: str, image_file_name: str) -> str:
    return IMAGE_URL_TEMPLATE.format(
        datastore_directory_url=datastore_directory_url, image_file_name=image_file_name
    )


class ImageGenerationPredictor(PredictWrapper):
    def __init__(self, model_uri, task_type, device=None):
        super().__init__(model_uri, task_type, device)

        self.prompt_column_name = "prompt"
        self.image_column_name = "generated_image"

    def predict(self, X_test, **kwargs):
        logger.info("z1 {} {}".format(type(X_test), X_test))

        # Get the datastore and the directory within it where the images will be uploaded.
        datastore = get_default_datastore()
        datastore_directory_name, datastore_directory_url = get_datastore_image_directory_name(datastore.name)

        logger.info("z2 {} {} {}".format(datastore.name, datastore_directory_name, datastore_directory_url))

        ys = []

        # Make the local temporary directory where the images will be saved.
        with tempfile.TemporaryDirectory() as temporary_directory_name:
            batch_size = 1
            image_counter = 0
            for idx in range(0, len(X_test), batch_size):
                y_batch = self.model.predict(X_test.iloc[idx : idx + batch_size])

                logger.info("z3")

                for i, image_str in enumerate(y_batch[self.image_column_name]):
                    if i == 0:
                        logger.info("z4 {}".format(type(image_str)))

                    # Save the current image to the local temporary directory.
                    image_file_name = get_image_file_name(image_counter=image_counter)
                    image_file_path = os.path.join(temporary_directory_name, image_file_name)
                    Image.open(io.BytesIO(base64.b64decode(image_str))).save(image_file_path)
                    if i == 0:
                        logger.info("z5")

                    image_url = get_image_url(datastore_directory_url, image_file_name)
                    ys.append(image_url)
                    if i == 0:
                        logger.info("z6 {}".format(image_url))

                    image_counter += 1

            # Upload the images to datastore.
            datastore.upload(src_dir=temporary_directory_name, target_path=datastore_directory_name)

        logger.info("z7 {}".format(ys[0]))

        return ys

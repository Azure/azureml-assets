# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Image generation predictor."""

from logging_utilities import get_logger
from task_factory.base import PredictWrapper
from task_factory.image.constants import ImagePredictionsLiterals
from task_factory.image.image_uploader import ImageUploader


logger = get_logger(name=__name__)


class ImageGenerationPredictor(PredictWrapper):
    def __init__(self, model_uri, task_type, device=None):
        super().__init__(model_uri, task_type, device)

        # TODO: use model parameter to get the name of the generated image.
        self.image_column_name = "generated_image"

    def predict(self, X_test, **kwargs):
        batch_size = kwargs.get(ImagePredictionsLiterals.BATCH_SIZE, 1)

        with ImageUploader() as image_uploader:
            for idx in range(0, len(X_test), batch_size):
                image_batch = self.model.predict(X_test.iloc[idx : idx + batch_size])

                for image_str in image_batch[self.image_column_name]:
                    image_uploader.upload_image(image_str)

        return image_uploader.urls

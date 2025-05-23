# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Image generation predictor."""

from typing import List

from logging_utilities import get_logger
from task_factory.base import PredictWrapper
from task_factory.image.constants import ImagePredictionsLiterals
from task_factory.image.image_uploader import ImageUploader


logger = get_logger(name=__name__)


class ImageGenerationPredictor(PredictWrapper):
    """Predictor for image generation tasks.

    Calls image generation model on text prompts and saves images in default datastore, returning their URLs.
    """

    def __init__(self, model_uri, task_type, device=None):
        """Initialize `ImageGenerationPredictor` members."""
        # Delegate to `PredictWrapper` constructor.
        super().__init__(model_uri, task_type, device)

        # Set the name of the output field with the generated image.
        # TODO: use model parameter to get the output image field.
        self.image_key = ImagePredictionsLiterals.GENERATED_IMAGE

    def predict(self, X_test, **kwargs) -> List[str]:
        """Generate images based on text prompts and save them.

        Args:
            X_test (pd.DataFrame): Input text prompts as a Pandas DataFrame with a "prompt" column name.
        Returns:
            List[str]: List of URLs for the generated images.
        """
        # Get the batch size if specified, else set to default 1.
        batch_size = kwargs.get(ImagePredictionsLiterals.BATCH_SIZE, 1)

        # Get relevant parameters if specified.
        predict_args = {name: kwargs[name] for name in [ImagePredictionsLiterals.GUIDANCE_SCALE] if name in kwargs}

        with ImageUploader() as image_uploader:
            # Go through prompts batch by batch.
            for idx in range(0, len(X_test), batch_size):
                # Run model prediction on current batch.
                image_batch = self.model.predict(X_test.iloc[idx:(idx + batch_size)], **predict_args)

                # Save batch of generated images in the default datastore.
                for encoded_image_bytes in image_batch[self.image_key]:
                    image_uploader.upload(encoded_image_bytes)

        return image_uploader.urls

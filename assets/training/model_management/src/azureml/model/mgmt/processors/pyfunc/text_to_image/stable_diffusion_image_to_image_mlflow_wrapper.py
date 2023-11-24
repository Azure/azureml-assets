# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Contains MLFlow pyfunc wrapper for stable diffusion image to image models.

Has methods to load the model and predict.
"""

import logging
import mlflow
import os
import pandas as pd
from diffusers import AutoPipelineForImage2Image
from config import (
    MLflowSchemaLiterals,
    Tasks,
    MLflowLiterals,
    BatchConstants,
    DatatypeLiterals,
)
from vision_utils import (
    get_pil_image,
    process_image_pandas_series,
    get_current_device,
    image_to_base64,
)

logger = logging.getLogger(__name__)


class StableDiffusionImageTexttoImageMLflowWrapper(mlflow.pyfunc.PythonModel):
    """MLflow model wrapper for stable diffusion image to image models."""

    def __init__(
        self,
        task_type: str,
    ) -> None:
        """Initialize model parameters for converting Huggingface Stable Diffusion image to image model to mlflow.

        :param task_type: Task type used in training.
        :type task_type: str
        """
        super().__init__()
        self._pipe = None
        self._task_type = task_type
        self._batch_output_folder = None

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        """
        Load a MLflow model with pyfunc.load_model().

        :param context: MLflow context containing artifacts that the model can use for inference
        :type context: mlflow.pyfunc.PythonModelContext
        """
        self._batch_output_folder = os.getenv(
            BatchConstants.BATCH_OUTPUT_PATH, default=False
        )

        if self._task_type == Tasks.IMAGE_TEXT_TO_IMAGE.value:
            try:
                model_dir = context.artifacts[MLflowLiterals.MODEL_DIR]
                self._pipe = AutoPipelineForImage2Image.from_pretrained(model_dir)
                self._pipe.to(get_current_device())
                logger.info("Model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load the the model. {str(e)}")
                raise
        else:
            raise ValueError(f"invalid task type {self._task_type}")

    def predict(
        self, context: mlflow.pyfunc.PythonModelContext, input_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Perform inference on the input data.

        :param context: MLflow context containing artifacts that the model can use for inference
        :type context: mlflow.pyfunc.PythonModelContext
        :param input_data: Pandas DataFrame with 2 columns named "prompt", "image" having text
                           input, initial image.
        :type input_data: pd.DataFrame
        :return: Pandas dataframe with corresponding generated images and NSFW flag.
                 Images in form of base64 string.
        :rtype: pd.DataFrame
        """
        # Decode the base64 image column
        images = input_data.loc[:, [MLflowSchemaLiterals.INPUT_COLUMN_IMAGE]].apply(
            axis=1, func=process_image_pandas_series
        )
        images = images.loc[:, 0].apply(func=get_pil_image).tolist()

        text_prompts = input_data.loc[
            :, MLflowSchemaLiterals.INPUT_COLUMN_PROMPT
        ].tolist()

        assert len(text_prompts) == len(images), (
            f"Invalid input. Number of text prompt, image are expected to be same. "
            f"But, found text prompt length {len(text_prompts)}, image length {len(images)}. "
        )

        generated_images = []
        try:
            outputs = self._pipe(
                prompt=text_prompts,
                image=images,
                return_dict=True,
            )
        except Exception as e:
            logger.error(f"Failed while running inference. {str(e)}")
            raise

        for img in outputs.images:
            generated_images.append(
                image_to_base64(img, format=DatatypeLiterals.IMAGE_FORMAT)
            )

        nsfw_content = None

        df = pd.DataFrame(
            {
                MLflowSchemaLiterals.OUTPUT_COLUMN_IMAGE: generated_images,
                MLflowSchemaLiterals.OUTPUT_COLUMN_NSFW_FLAG: nsfw_content,
            }
        )

        return df

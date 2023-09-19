# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Contains MLFlow pyfunc wrapper for stable diffusion inpainting models.

Has methods to load the model and predict.
"""

from diffusers import StableDiffusionInpaintPipeline
import mlflow
import os
import pandas as pd
import torch

from config import MLflowSchemaLiterals, Tasks, MLflowLiterals, BatchConstants
from utils import image_to_str, save_image


class StableDiffusionInpaintingMLflowWrapper(mlflow.pyfunc.PythonModel):
    """MLflow model wrapper for stable diffusion inpainting models."""

    def __init__(
        self,
        task_type: str,
    ) -> None:
        """Initialize model parameters for converting Huggingface Stable Diffusion inpainting model to mlflow.

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

        if self._task_type == Tasks.TEXT_TO_IMAGE_INPAINTING.value:
            try:
                _map_location = "cuda" if torch.cuda.is_available() else "cpu"
                model_dir = context.artifacts[MLflowLiterals.MODEL_DIR]
                self._pipe = StableDiffusionInpaintPipeline.from_pretrained(model_dir)
                self._pipe.to(_map_location)
                print("Model loaded successfully")
            except Exception as e:
                print("Failed to load the the model.")
                print(e)
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
        :param input_data: Pandas DataFrame with a column name ["prompt"] having text
                           input for which image has to be generated.
        :type input_data: pd.DataFrame
        :return: Pandas dataframe with input text prompts, their corresponding generated images and NSFW flag.
                 Images in form of base64 string.
        :rtype: pd.DataFrame
        """
        text_prompts = input_data[MLflowSchemaLiterals.INPUT_COLUMN_PROMPT].tolist()
        image = input_data[MLflowSchemaLiterals.INPUT_COLUMN_IMAGE].tolist()
        mask_image = input_data[MLflowSchemaLiterals.INPUT_COLUMN_MASK_IMAGE].tolist()

        assert len(text_prompts) == len(image) == len(mask_image), (
            f"Invalid input. Number of text prompt, image and mask image are expected to be same. "
            f"But, found text prompt length {len(text_prompts)}, image length {len(image)} and "
            f"mask_image length {len(mask_image)}"
        )

        generated_images = []
        nsfw_content = []
        if self._batch_output_folder:
            # Batch endpoint
            for text_prompt, image, mask_image in zip(text_prompts, image, mask_image):
                output = self._pipe(
                    prompt=text_prompt,
                    image=image,
                    mask_image=mask_image,
                    return_dict=True,
                )

                # Save image in batch output folder and append the image file name to generated_images list
                generated_images.append(
                    save_image(self._batch_output_folder, output.images[0])
                )
                nsfw_content.append(
                    output.nsfw_content_detected[0]
                    if output.nsfw_content_detected
                    else None
                )
        else:
            # Online endpoint
            outputs = self._pipe(
                prompt=text_prompts,
                image=image,
                mask_image=mask_image,
                return_dict=True,
            )

            for img in outputs.images:
                generated_images.append(image_to_str(img))

            nsfw_content = outputs.nsfw_content_detected

        df = pd.DataFrame(
            {
                MLflowSchemaLiterals.OUTPUT_COLUMN_IMAGE: generated_images,
                MLflowSchemaLiterals.OUTPUT_COLUMN_NSFW_FLAG: output.nsfw_content_detected,
            }
        )

        return df

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Contains MLFlow pyfunc wrapper for stable diffusion models.

Has methods to load the model and predict.
"""

from diffusers import StableDiffusionPipeline, DiffusionPipeline
import mlflow
import os
import pandas as pd
import torch

from config import (
    MLflowSchemaLiterals, Tasks, MLflowLiterals,
    BatchConstants, DatatypeLiterals,
    SupportedTextToImageModelFamilyPipelines
)
from vision_utils import image_to_base64


class StableDiffusionMLflowWrapper(mlflow.pyfunc.PythonModel):
    """MLflow model wrapper for stable diffusion models."""

    def __init__(
        self,
        task_type: str,
        model_family: str
    ) -> None:
        """Initialize model parameters for converting Huggingface StableDifusion model to mlflow.

        :param task_type: Task type used in training.
        :type task_type: str
        :param model_family: Model family of the stable diffusion task.
        :type model_family: str
        """
        super().__init__()
        self._pipe = None
        self._task_type = task_type
        self.batch_output_folder = None
        self._model_family = model_family

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        """
        Load a MLflow model with pyfunc.load_model().

        :param context: MLflow context containing artifacts that the model can use for inference
        :type context: mlflow.pyfunc.PythonModelContext
        """
        self.batch_output_folder = os.getenv(
            BatchConstants.BATCH_OUTPUT_PATH, default=False
        )

        if self._task_type == Tasks.TEXT_TO_IMAGE.value:
            try:
                _map_location = "cuda" if torch.cuda.is_available() else "cpu"
                model_dir = context.artifacts[MLflowLiterals.MODEL_DIR]
                if self._model_family == SupportedTextToImageModelFamilyPipelines.DECI_DIFFUSION.value:
                    self._pipe = StableDiffusionPipeline.from_pretrained(model_dir, custom_pipeline=model_dir)
                    self._pipe.unet = self._pipe.unet.from_pretrained(model_dir, subfolder='flexible_unet')
                elif self._model_family == SupportedTextToImageModelFamilyPipelines.STABLE_DIFFUSION_XL.value:
                    self._pipe = DiffusionPipeline.from_pretrained(model_dir, custom_pipeline=model_dir)
                else:
                    self._pipe = StableDiffusionPipeline.from_pretrained(model_dir)

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

        output = self._pipe(text_prompts)
        generated_images = []
        for img in output.images:
            generated_images.append(image_to_base64(img, format=DatatypeLiterals.IMAGE_FORMAT))

        df = pd.DataFrame(
            {
                MLflowSchemaLiterals.INPUT_COLUMN_PROMPT: text_prompts,
                MLflowSchemaLiterals.OUTPUT_COLUMN_IMAGE: generated_images,
                MLflowSchemaLiterals.OUTPUT_COLUMN_NSFW_FLAG: output.nsfw_content_detected,
            }
        )

        return df

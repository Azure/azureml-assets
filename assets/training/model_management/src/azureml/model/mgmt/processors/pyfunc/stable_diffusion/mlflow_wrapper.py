# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Contains MLFlow pyfunc wrapper for stable diffusion models.
Has methods to load the model and predict.
"""

import base64
from diffusers import StableDiffusionPipeline
import pandas as pd
from io import BytesIO
import mlflow
import torch

from .constants import COLUMN_NAMES, DATATYPE_LITERALS, Tasks


class StableDiffusionMLflowWrapper(mlflow.pyfunc.PythonModel):
    """MLflow model wrapper for stable diffusion models."""

    def __init__(
            self,
            task_type: str,
            model_id: str,
    ) -> None:
        """Constructor for MLflow wrapper class

        :param task_type: Task type used in training.
        :type task_type: str
        :param model_id: Hugging face model id of stable diffusion models.
        :type model_id: str
        """
        super().__init__()
        self._pipe = None
        self._task_type = task_type
        self._model_id = model_id

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        """
        Load a MLflow model with pyfunc.load_model().

        :param context: MLflow context containing artifacts that the model can use for inference
        :type context: mlflow.pyfunc.PythonModelContext
        """
        if self._task_type in [Tasks.TEXT_TO_IMAGE, Tasks.IMAGE_INPAINTING]:
            try:
                _map_location = "cuda" if torch.cuda.is_available() else "cpu"

                self._pipe = StableDiffusionPipeline.from_pretrained(self._model_id)
                self._pipe.to(_map_location)
                print("Model loaded successfully")
            except Exception:
                print("Failed to load the the model.")
                raise
        else:
            raise ValueError(f"invalid task type {self._task_type}")

    def predict(self, context: mlflow.pyfunc.PythonModelContext, input_data: pd.DataFrame) -> pd.DataFrame:
        """
        Perform inference on the input data.

        :param context: MLflow context containing artifacts that the model can use for inference
        :type context: mlflow.pyfunc.PythonModelContext
        :param input_data: Pandas DataFrame with a column name ["test_prompts"] having text
                           input for which image has to be generated.
        :type input_data: pd.DataFrame
        :return: Pandas dataframe with input test_prompts and its corresponding generated images.
                 Images in form of base64 string.
        :rtype: pd.DataFrame
        """
        print("input_data received: " + str(input_data))
        text_prompts = input_data[COLUMN_NAMES.TEXT_PROMPTS].tolist()
        images = self._pipe(text_prompts).images
        generated_images = []
        for img in images:
            buffered = BytesIO()
            img.save(buffered, format=DATATYPE_LITERALS.IMAGE_FORMAT)
            img_str = base64.b64encode(buffered.getvalue()).decode(DATATYPE_LITERALS.STR_ENCODING)
            generated_images.append(img_str)

        print("Image generation successful")
        df = pd.DataFrame({COLUMN_NAMES.TEXT_PROMPTS: text_prompts, COLUMN_NAMES.GENERATED_IMAGES: generated_images})

        return df

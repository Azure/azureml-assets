# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Contains MLFlow pyfunc wrapper for stable diffusion models.
Has methods to load the model and predict.
"""

import base64
from diffusers import StableDiffusionPipeline
from io import BytesIO
import mlflow
import pandas as pd
import torch

from .constants import ColumnNames, DatatypeLiterals, MLflowLiterals, Tasks


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
        :param model_id: Hugging face model id corresponding to stable diffusion models supported by AML.
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
        if self._task_type == Tasks.TEXT_TO_IMAGE:
            try:
                _map_location = "cuda" if torch.cuda.is_available() else "cpu"
                model_dir = context.artifacts[MLflowLiterals.MODEL_DIR]
                self._pipe = StableDiffusionPipeline.from_pretrained(model_dir)
                self._pipe.to(_map_location)
                print("Model loaded successfully")
            except Exception as e:
                print("Failed to load the the model.")
                print(e)
                raise
        else:
            raise ValueError(f"invalid task type {self._task_type}")

    def predict(self, context: mlflow.pyfunc.PythonModelContext, input_data: pd.DataFrame) -> pd.DataFrame:
        """
        Perform inference on the input data.

        :param context: MLflow context containing artifacts that the model can use for inference
        :type context: mlflow.pyfunc.PythonModelContext
        :param input_data: Pandas DataFrame with a column name ["text_prompt"] having text
                           input for which image has to be generated.
        :type input_data: pd.DataFrame
        :return: Pandas dataframe with input text_prompts and its corresponding generated images.
                 Images in form of base64 string.
        :rtype: pd.DataFrame
        """
        text_prompts = input_data[ColumnNames.TEXT_PROMPT].tolist()
        output = self._pipe(text_prompts)

        generated_images = []
        for img in output.images:
            buffered = BytesIO()
            img.save(buffered, format=DatatypeLiterals.IMAGE_FORMAT)
            img_str = base64.b64encode(buffered.getvalue()).decode(DatatypeLiterals.STR_ENCODING)
            generated_images.append(img_str)

        print("Image generation successful")
        print(f"At least one NSFW image detected: {any(output.nsfw_content_detected)}")

        df = pd.DataFrame({ColumnNames.TEXT_PROMPT: text_prompts,
                           ColumnNames.GENERATED_IMAGE: generated_images,
                           ColumnNames.NSFW_FLAG: output.nsfw_content_detected
                           }
                          )

        return df

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Contains MLFlow pyfunc wrapper for stable diffusion models.

Has methods to load the model and predict.
"""

from diffusers import StableDiffusionPipeline
import mlflow
import os
import pandas as pd
import torch

from constants import BatchConstants, ColumnNames, MLflowLiterals, Tasks
from utils import image_to_str, save_image

class StableDiffusionMLflowWrapper(mlflow.pyfunc.PythonModel):
    """MLflow model wrapper for stable diffusion models."""

    def __init__(
            self,
            task_type: str,
    ) -> None:
        """Initialize model parameters for converting Huggingface StableDifusion model to mlflow.

        :param task_type: Task type used in training.
        :type task_type: str
        """
        super().__init__()
        self._pipe = None
        self._task_type = task_type
        self.batch_output_folder = None

    def predict_batch(self, text_prompts):
        """
        For batch we are trying to do prediction in a loop instead of sending complete list at once.
        Sending a big list of text prompts at once might lead to out of memory related errors.

        :param text_prompts: A list of text prompts for which we need to generate images.
        :type text_prompts: list
        :return: Pandas dataframe having generated images and NSFW flag. Images in form of base64 string.
        :rtype: pandas.DataFrame
        """
        generated_images = []
        nsfw_content_detected = []
        for text_prompt in text_prompts:
            output = self._pipe(text_prompt)
            img = output.images[0]
            generated_images.append(save_image(self.batch_output_folder, img))
            nsfw_content_detected.append(output.nsfw_content_detected)

        print("All images generated in a loop.")

        df = pd.DataFrame({ColumnNames.TEXT_PROMPT: text_prompts,
                           ColumnNames.GENERATED_IMAGE: generated_images,
                           ColumnNames.NSFW_FLAG: nsfw_content_detected
                           }
                          )

        return df

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        """
        Load a MLflow model with pyfunc.load_model().

        :param context: MLflow context containing artifacts that the model can use for inference
        :type context: mlflow.pyfunc.PythonModelContext
        """
        self.batch_output_folder = os.getenv(BatchConstants.BATCH_OUTPUT_PATH, default=False)

        if self._task_type == Tasks.TEXT_TO_IMAGE.value:
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
        :param input_data: Pandas DataFrame with a column name ["prompt"] having text
                           input for which image has to be generated.
        :type input_data: pd.DataFrame
        :return: Pandas dataframe with input text prompts, their corresponding generated images and NSFW flag.
                 Images in form of base64 string.
        :rtype: pd.DataFrame
        """
        text_prompts = input_data[ColumnNames.TEXT_PROMPT].tolist()

        if self.batch_output_folder:
            # Batch endpoint
            return self.predict_batch(text_prompts)
        else:
            # Online endpoint
            output = self._pipe(text_prompts)
            generated_images = []
            for img in output.images:
                generated_images.append(image_to_str(img))

            print("Image generation successful")

            df = pd.DataFrame({ColumnNames.TEXT_PROMPT: text_prompts,
                               ColumnNames.GENERATED_IMAGE: generated_images,
                               ColumnNames.NSFW_FLAG: output.nsfw_content_detected
                               }
                              )

            return df

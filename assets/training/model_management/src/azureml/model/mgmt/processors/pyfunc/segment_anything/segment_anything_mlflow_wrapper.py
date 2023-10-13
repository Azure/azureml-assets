# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Contains MLFlow pyfunc wrapper for stable diffusion models.

Has methods to load the model and predict.
"""

import mlflow
import io
import pandas as pd
import torch
from transformers import SamModel, SamProcessor
from PIL import Image
from config import MLflowSchemaLiterals, Tasks, MLflowLiterals


class SegmentAnythingDiffusionMLflowWrapper(mlflow.pyfunc.PythonModel):
    """MLflow model wrapper for segment anything models."""

    def __init__(
        self,
        task_type: str,
    ) -> None:
        """Initialize model parameters for converting Huggingface StableDifusion model to mlflow.

        :param task_type: Task type used in training.
        :type task_type: str
        """
        super().__init__()
        self._task_type = task_type

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        """
        Load a MLflow model with pyfunc.load_model().

        :param context: MLflow context containing artifacts that the model can use for inference
        :type context: mlflow.pyfunc.PythonModelContext
        """

        if self._task_type == Tasks.SEGMENT_ANYTHING.value:
            try:
                _map_location = "cuda" if torch.cuda.is_available() else "cpu"
                model_dir = context.artifacts[MLflowLiterals.MODEL_DIR]
                self._model = SamModel.from_pretrained(model_dir).to(_map_location)
                self._processor = SamProcessor.from_pretrained(model_dir)
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
        :param input_data: Pandas DataFrame with a column name ["image"], ["input_boxes"], ["labels] having list of list of list encoded to string.
        :type input_data: pd.DataFrame
        :return: Pandas dataframe with input text prompts, their corresponding generated images and NSFW flag.
                 Images in form of base64 string.
        :rtype: pd.DataFrame
        """
        from vision_utils import process_image, string_to_nested_float_list

        # Do inference one input at a time.
        response = []
        for image, input_points, input_boxes, input_labels in zip(
            input_data[MLflowSchemaLiterals.INPUT_COLUMN_IMAGE],
            input_data[MLflowSchemaLiterals.INPUT_COLUMN_INPUT_POINTS],
            input_data[MLflowSchemaLiterals.INPUT_COLUMN_INPUT_BOXES],
            input_data[MLflowSchemaLiterals.INPUT_COLUMN_INPUT_LABELS],
        ):
            
            # Decode the image and make a PIL Image object.
            pil_image = Image.open(io.BytesIO(process_image(image)))

            if input_points:
                # conver the string to list of list of list of integers/float
                input_points = string_to_nested_float_list(input_points)
            else:
                input_points = None
            
            if input_boxes:
                # conver the string to list of list of list of integers/float
                input_boxes = string_to_nested_float_list(input_boxes)
            else:
                input_boxes = None

            if input_labels:
                # conver the string to list of list of list of integers/float
                input_labels = string_to_nested_float_list(input_labels)
            else:
                input_labels = None

            # Do inference.
            _map_location = "cuda" if torch.cuda.is_available() else "cpu"
            processed_inputs = self._processor(pil_image, input_boxes=input_boxes, input_points=input_points, input_labels=input_labels, return_tensors="pt").to(_map_location)
            with torch.no_grad():
                outputs = self._model(**processed_inputs)

            masks = self._processor.image_processor.post_process_masks(outputs.pred_masks.to(_map_location), processed_inputs["original_sizes"].to(_map_location), processed_inputs["reshaped_input_sizes"].to(_map_location))
            scores = outputs.iou_scores

            # convert the tensors to list
            masks_list = masks[0].tolist()
            scores_list = scores.squeeze(dim=0).tolist()

            # prepare the response dataframe
            pred = {"masks" : masks_list, "scores": scores_list}
            response.append(pred)

        df_responses = pd.DataFrame({MLflowSchemaLiterals.OUTPUT_COLUMN_RESPONSE: response})
        return df_responses
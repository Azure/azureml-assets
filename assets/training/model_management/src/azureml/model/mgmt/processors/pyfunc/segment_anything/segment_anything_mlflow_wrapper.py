# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Contains MLFlow pyfunc wrapper for stable diffusion models.

Has methods to load the model and predict.
"""

import mlflow
import os
import io
import pandas as pd
import torch
import torch
from transformers import SamModel, SamProcessor
from PIL import Image
from config import MLflowSchemaLiterals, Tasks, MLflowLiterals, BatchConstants


def polygon_cal(numpy_array):
    from masktools import convert_mask_to_polygon

    # Initialize a list to store polygons for each batch
    all_polygons = []

    batch_size, num_channels, height, width = numpy_array.shape
    try:
        # Loop through batches
        for batch_index in range(batch_size):
            # Initialize a list to store polygons for each channel in the current batch
            batch_polygons = []

            # Loop through channels
            for channel_index in range(num_channels):
                channel = numpy_array[batch_index, channel_index, :, :]
                polygon = convert_mask_to_polygon(channel)
                batch_polygons.append(polygon)

            # Append the list of polygons for the current batch to the overall list
            all_polygons.append(batch_polygons)
        print("Successfully converted the mask to polygon.")
    except Exception as e:
        print("Failed to convert the mask to polygon.")
        print(e)
        raise
    return all_polygons


class SegmentAnythingDiffusionMLflowWrapper(mlflow.pyfunc.PythonModel):
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
        self._task_type = task_type
        self.batch_output_folder = None

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        """
        Load a MLflow model with pyfunc.load_model().

        :param context: MLflow context containing artifacts that the model can use for inference
        :type context: mlflow.pyfunc.PythonModelContext
        """
        self.batch_output_folder = os.getenv(
            BatchConstants.BATCH_OUTPUT_PATH, default=False
        )
        

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

            # convert the tensors to numpy array for mask to polygon conversion
            masks_numpy = masks[0].cpu().numpy()
            scores_numpy = scores.squeeze(dim=0).cpu().numpy()

            # convert the numpy binary mask to polygon
            polygons = polygon_cal(masks_numpy)

            # prepare the response dataframe
            pred = {"masks" : []}

            for i in range(len(polygons)):
                list_of_mask = []
                for j in range(len(polygons[i])):
                    mask = {"polygon" : polygons[i][j], "iou_score" : scores_numpy[i][j]}
                    list_of_mask.append(mask)
                pred["masks"].append(list_of_mask)

            df_responses = pd.DataFrame(pred)
            return df_responses
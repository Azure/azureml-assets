# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""MLflow PythonModel wrapper class that loads the MLflow model, preprocess inputs and performs inference."""

import mlflow
from PIL import Image
import pandas as pd
import tempfile

from transformers import AutoProcessor, Blip2ForConditionalGeneration
from config import MLflowSchemaLiterals, MLflowLiterals, Tasks
from typing import List

try:
    # Use try/except since vision_utils is added as part of model export and not available when initializing
    # model wrapper for save_model().
    from vision_utils import create_temp_file, process_image_pandas_series, get_current_device
except ImportError:
    pass


class BLIP2MLFlowModelWrapper(mlflow.pyfunc.PythonModel):
    """MLflow model wrapper for BLIP2 model."""

    def __init__(
        self,
        task_type: str,
    ) -> None:
        """Initialize MLflow wrapper class.

        :param task_type: Task type used in training.
        :type task_type: str
        """
        super().__init__()
        self._processor = None
        self._model = None
        self._task_type = task_type

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        """
        Load a MLflow model with pyfunc.load_model().

        :param context: MLflow context containing artifacts that the model can use for inference
        :type context: mlflow.pyfunc.PythonModelContext
        """
        if self._task_type == (Tasks.IMAGE_TO_TEXT.value):
            try:
                model_dir = context.artifacts[MLflowLiterals.MODEL_DIR]
                self._processor = AutoProcessor.from_pretrained(model_dir)
                self._model = Blip2ForConditionalGeneration.from_pretrained(
                    model_dir)
                self._device = get_current_device()
                self._model.to(self._device)

                print("Model loaded successfully")
            except Exception as e:
                print("Failed to load the the model.")
                print(e)
                raise
        else:
            raise ValueError(f"invalid task type {self._task_type}")

    def predict(self, context: mlflow.pyfunc.PythonModelContext, input_data: pd.DataFrame) -> pd.DataFrame:
        """Perform inference on the input data.

        :param context: MLflow context containing artifacts that the model can use for inference
        :type context: mlflow.pyfunc.PythonModelContext
        :param input_data: Input images for prediction.
        :type input_data: Pandas DataFrame with a first column name ["image"] of images where each
        image is in base64 string format or url to the image.
        :return: Output of inferencing
        :rtype: Pandas DataFrame with columns ["text"]
        """
        # Decode the base64 image column
        decoded_images = input_data.loc[
            :, [MLflowSchemaLiterals.INPUT_COLUMN_IMAGE]
        ].apply(axis=1, func=process_image_pandas_series)

        with tempfile.TemporaryDirectory() as tmp_output_dir:
            image_path_list = (
                decoded_images.iloc[:, 0]
                .map(lambda row: create_temp_file(row, tmp_output_dir)[0])
                .tolist()
            )

            generated_text_list = self.run_inference_batch(
                image_path_list=image_path_list,
            )

        df_result = pd.DataFrame(
            columns=[
                MLflowSchemaLiterals.OUTPUT_COLUMN_TEXT
            ]
        )

        df_result[MLflowSchemaLiterals.OUTPUT_COLUMN_TEXT] = (
            generated_text_list)
        return df_result

    def run_inference_batch(
        self,
        image_path_list: List
    ) -> List[str]:
        """Perform inference on batch of input images.

        :param image_path_list: list of image paths for inferencing.
        :type image_path_list: List
        :return: List of generated texts
        :rtype: List of strings
        """
        image_list = [Image.open(img_path) for img_path in image_path_list]

        inputs = self._processor(images=image_list,
                                 return_tensors="pt").to(self._device)
        generated_ids = self._model.generate(**inputs)
        generated_text_list = self._processor.batch_decode(generated_ids, skip_special_tokens=True)

        text_list = [t.strip() for t in generated_text_list]
        return text_list

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""MLflow PythonModel wrapper class that loads the MLflow model, preprocess inputs and performs inference."""

import io

import mlflow
import pandas as pd

from PIL import Image
from transformers import (AutoProcessor, BlipForConditionalGeneration,
                          BlipForQuestionAnswering, Blip2ForConditionalGeneration)

from config import MLflowSchemaLiterals, MLflowLiterals, Tasks, HfBlipModelId


try:
    # Use try/except since vision_utils is added as part of model export and not available when initializing
    # model wrapper for save_model().
    from vision_utils import process_image, get_current_device
except ImportError:
    pass


class BLIPMLFlowModelWrapper(mlflow.pyfunc.PythonModel):
    """MLflow model wrapper for BLIP model family."""

    def __init__(
        self,
        task_type: str,
        model_id: str
    ) -> None:
        """Initialize MLflow wrapper class.

        :param task_type: Task type used in training.
        :type task_type: str
        :param model_id: HF model id for BLIP family.
        :type model_id: str
        """
        super().__init__()
        self._processor = None
        self._model = None
        self._task_type = task_type
        self._model_id = model_id

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        """
        Load a MLflow model with pyfunc.load_model().

        :param context: MLflow context containing artifacts that the model can use for inference
        :type context: mlflow.pyfunc.PythonModelContext
        """
        if self._task_type in [Tasks.IMAGE_TO_TEXT.value, Tasks.VISUAL_QUESTION_ANSWERING.value]:
            try:
                model_dir = context.artifacts[MLflowLiterals.MODEL_DIR]
                self._processor = AutoProcessor.from_pretrained(model_dir)

                if self._model_id == HfBlipModelId.BLIP_IMAGE_TO_TEXT.value:
                    self._model = BlipForConditionalGeneration.from_pretrained(
                        model_dir)
                elif self._model_id == HfBlipModelId.BLIP_VQA.value:
                    self._model = BlipForQuestionAnswering.from_pretrained(
                        model_dir)
                elif self._model_id in [HfBlipModelId.BLIP2.value]:
                    self._model = Blip2ForConditionalGeneration.from_pretrained(
                        model_dir)
                else:
                    raise ValueError(f"invalid model id {self._model_id}")

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
        # Read all the input images.
        pil_images = [
            Image.open(io.BytesIO(process_image(image)))
            for image in input_data[MLflowSchemaLiterals.INPUT_COLUMN_IMAGE]
        ]

        # Preprocess the image and text data (if necessary) and move it to the device.
        if self._task_type == Tasks.IMAGE_TO_TEXT.value:
            inputs = self._processor(images=pil_images, return_tensors="pt")
        elif self._task_type == Tasks.VISUAL_QUESTION_ANSWERING.value:
            inputs = self._processor(
                images=pil_images, text=input_data[MLflowSchemaLiterals.INPUT_COLUMN_TEXT].tolist(),
                padding=True, return_tensors="pt",
            )
        else:
            raise ValueError(f"invalid task type {self._task_type}")
        inputs = inputs.to(self._device)

        # Do inference and convert to human readable strings.
        generated_ids = self._model.generate(**inputs)
        generated_text_list = self._processor.batch_decode(generated_ids, skip_special_tokens=True)
        generated_text_list = [t.strip() for t in generated_text_list]

        # Convert to Pandas dataframe and return.
        df_result = pd.DataFrame(
            {
                MLflowSchemaLiterals.OUTPUT_COLUMN_TEXT: generated_text_list
            }
        )
        return df_result

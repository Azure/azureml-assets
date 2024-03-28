# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""MLflow PythonModel wrapper class that loads the MLflow model, preprocess inputs and performs inference."""

import io

import mlflow
import pandas as pd

from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification

from config import MLflowSchemaLiterals, MLflowLiterals, Tasks


try:
    # Use try/except since vision_utils is added as part of model export and not available when initializing
    # model wrapper for save_model().
    from vision_utils import process_image, get_current_device
except ImportError:
    pass


class CLIPMLFlowModelWrapper(mlflow.pyfunc.PythonModel):
    """MLflow model wrapper for CLIP model."""

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
        self._supported_task = Tasks.ZERO_SHOT_IMAGE_CLASSIFICATION.value

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        """
        Load a MLflow model with pyfunc.load_model().

        :param context: MLflow context containing artifacts that the model can use for inference
        :type context: mlflow.pyfunc.PythonModelContext
        """
        if self._task_type == self._supported_task:
            try:
                model_dir = context.artifacts[MLflowLiterals.MODEL_DIR]
                self._processor = AutoProcessor.from_pretrained(model_dir)
                self._model = AutoModelForZeroShotImageClassification.from_pretrained(model_dir)
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
        :param input_data: Input images for prediction and candidate labels.
        :type input_data: Pandas DataFrame with a first column name ["image"] of images where each
        image is in base64 String format, and second column name ["text"] where the first row contains the
        candidate labels and the remaining rows are ignored.
        :return: Output of inferencing
        :rtype: Pandas DataFrame with columns ["probs", "labels"]
        """
        # Read all the input images.
        pil_images = [
            Image.open(io.BytesIO(process_image(image)))
            for image in input_data[MLflowSchemaLiterals.INPUT_COLUMN_IMAGE]
        ]

        try:
            # Parse comma separated classes and remove leading and trailing whitespace.
            captions = map(
                str.strip,
                input_data[MLflowSchemaLiterals.INPUT_COLUMN_TEXT].iloc[0].split(',')
            )
            captions = list(filter(None, captions))  # remove any empty strings
            if len(captions) == 0:
                raise ValueError("No labels were provided")
        except Exception:
            raise ValueError(
                    "The provided labels cannot be parsed. The first row of the \"text\" column is expected "
                    "to contain a string with the comma-separated labels"
            )

        # Preprocess the image and text data and move it to the device.
        inputs = self._processor(text=captions, images=pil_images, return_tensors="pt", padding=True)
        inputs = inputs.to(self._device)

        # Do inference and compute the class probabilities.
        outputs = self._model(**inputs)
        logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities

        # Covert to Pandas dataframe and return.
        df_result = pd.DataFrame(
            {
                MLflowSchemaLiterals.OUTPUT_COLUMN_PROBS: probs.tolist(),
                MLflowSchemaLiterals.OUTPUT_COLUMN_LABELS: [captions] * len(probs),
            }
        )
        return df_result

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""MLflow PythonModel wrapper that loads the DinoV2 model, preprocesses inputs and performs embedding inference."""

import io

import mlflow
import pandas as pd

from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from config import MLflowSchemaLiterals, MLflowLiterals, Tasks


class DinoV2EmbeddingsMLFlowModelWrapper(mlflow.pyfunc.PythonModel):
    """MLflow model wrapper for DinoV2 embeddings model, used for getting feature embeddings."""

    def __init__(
        self,
        task_type: str,
    ) -> None:
        """Initialize wrapper class.

        :param task_type: Task type used for inference.
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
        from vision_utils import get_current_device

        if self._task_type == Tasks.EMBEDDINGS.value:
            try:
                model_dir = context.artifacts[MLflowLiterals.MODEL_DIR]
                self._processor = AutoImageProcessor.from_pretrained(model_dir)
                self._model = AutoModel.from_pretrained(model_dir)
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
        """Perform inference on the input images.

        Assumption: it is ok to crash if the user sends too many images, causing an OOM error. CLIP and BLIP
        assume the same.

        :param context: MLflow context containing artifacts that the model can use for inference
        :type context: mlflow.pyfunc.PythonModelContext
        :param input_data: input images, either as publicly accessible urls or base64 strings
        :type input_data: Pandas DataFrame with column "image"
        :return: image embeddings
        :rtype: Pandas DataFrame with column "image_features"
        """
        from vision_utils import process_image

        # Read all the input images.
        pil_images = [
            Image.open(io.BytesIO(process_image(image)))
            for image in input_data[MLflowSchemaLiterals.INPUT_COLUMN_IMAGE]
        ]

        # Do inference and extract the embeddings out of the output structure.
        inputs = self._processor(images=pil_images, return_tensors="pt")
        outputs = self._model(**{
            key: inputs[key].to(self._device) for key in inputs
        })
        embeddings = outputs.last_hidden_state[:, 0, :]

        # Offload to CPU, convert to Python list and accumulate into result embedding list.
        embeddings = embeddings.detach().cpu().numpy().tolist()

        # Convert embeddings to Pandas dataframe and return.
        df_responses = pd.DataFrame({MLflowSchemaLiterals.OUTPUT_COLUMN_IMAGE_FEATURES: embeddings})
        return df_responses

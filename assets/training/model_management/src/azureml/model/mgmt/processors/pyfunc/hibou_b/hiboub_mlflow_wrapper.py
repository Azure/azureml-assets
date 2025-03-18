# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""MLflow PythonModel wrapper that loads the Hibou-B image model and performs inference to return the pooler output."""

import io
import mlflow
import pandas as pd

from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from config import MLflowSchemaLiterals, MLflowLiterals

try:
    # Use try/except since vision_utils is added as part of model export and not available when initializing
    # model wrapper for save_model().
    from vision_utils import process_image, get_current_device
except ImportError:
    pass


class HibouBPoolerMLFlowModelWrapper(mlflow.pyfunc.PythonModel):
    """MLflow model wrapper for Hibou-B image model, used for obtaining pooler outputs."""

    def __init__(self, task_type: str) -> None:
        """
        Initialize the wrapper class.

        :param task_type: Task type used for inference (e.g., Tasks.EMBEDDINGS.value)
        :type task_type: str
        """
        super().__init__()
        self._task_type = task_type

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        """
        Load the model and processor from the model directory specified in MLflow artifacts.

        :param context: MLflow context containing artifacts for the model
        :type context: mlflow.pyfunc.PythonModelContext
        """
        try:
            model_dir = context.artifacts[MLflowLiterals.MODEL_DIR]
            self._processor = AutoImageProcessor.from_pretrained(
                model_dir, trust_remote_code=True
            )
            self._model = AutoModel.from_pretrained(model_dir, trust_remote_code=True)
            self._device = get_current_device()
            self._model.to(self._device)
            print("Hibou-B model loaded successfully.")
        except Exception as e:
            print("Failed to load the Hibou-B model.")
            print(e)
            raise

    def predict(
        self, context: mlflow.pyfunc.PythonModelContext, input_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Perform inference on the input images and return the pooler outputs.

        The input_data is expected to be a Pandas DataFrame with a column specified by
        MLflowSchemaLiterals.INPUT_COLUMN_IMAGE, containing image data as publicly accessible URLs,
        base64 strings, or file paths.

        :param context: MLflow context containing artifacts for the model
        :type context: mlflow.pyfunc.PythonModelContext
        :param input_data: input images in a DataFrame column.
        :type input_data: pd.DataFrame
        :return: A DataFrame with a column for pooler outputs
        :rtype: pd.DataFrame
        """
        pil_images = [
            Image.open(io.BytesIO(process_image(image)))
            for image in input_data[MLflowSchemaLiterals.INPUT_COLUMN_IMAGE]
        ]

        inputs = self._processor(images=pil_images, return_tensors="pt")
        inputs = {key: value.to(self._device) for key, value in inputs.items()}
        outputs = self._model(**inputs)

        # Extract pooler output; fallback to using the [CLS] token representation if needed.
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooler_output = outputs.pooler_output
        else:
            pooler_output = outputs.last_hidden_state[:, 0, :]
        pooler_output = pooler_output.detach().cpu().numpy().tolist()
        return pd.DataFrame(
            {MLflowSchemaLiterals.OUTPUT_COLUMN_IMAGE_FEATURES: pooler_output}
        )

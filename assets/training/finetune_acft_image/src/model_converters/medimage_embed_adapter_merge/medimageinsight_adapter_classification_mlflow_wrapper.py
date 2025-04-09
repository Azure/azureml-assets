# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""MLflow PythonModel wrapper class that loads the MLflow model, preprocess inputs and performs inference."""

import json
import os
import mlflow
from PIL import Image
import pandas as pd
import torch
import tempfile

from medimageinsight_mlflow_wrapper import MEDIMAGEINSIGHTMLFlowModelWrapper
from config import MLflowLiterals, MLflowSchemaLiterals
from typing import List, Tuple

import base64
from io import BytesIO

from adapter_model import MLP_model
import logging
logger = logging.getLogger(__name__)


class MEDIMAGEINSIGHTClassificationMLFlowModelWrapper(
    MEDIMAGEINSIGHTMLFlowModelWrapper
):
    """MLflow model wrapper for CLIP model, used for getting feature embeddings."""

    def __init__(
        self,
        task_type: str,
        vision_model_name: str,
        language_model_name: str,
    ) -> None:
        """Initialize MLflow wrapper class.

        :param task_type: Task type used in training.
        :type task_type: str
        """
        super().__init__(task_type, vision_model_name, language_model_name)
        self._supported_task = "image-classification"
        self.adapter_model = None

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        """Load the model and artifacts from the MLflow context.

        :param context: MLflow context containing artifacts that the model can use for inference
        :type context: mlflow.pyfunc.PythonModelContext
        """
        super().load_context(context)

        # also load the adapter model
        logger.info("Loading Adapter Model")
        model_dir = context.artifacts[MLflowLiterals.MODEL_DIR]
        model_path = os.path.join(model_dir, "adapter_model", "adapter_model.pth")
        config_path = os.path.join(model_dir, "adapter_model", "config.json")
        config_data = json.load(open(config_path))
        self.labels = config_data["labels"]
        self.adapter_model = MLP_model(
            in_channels=config_data["in_channels"],
            hidden_dim=config_data["hidden_dim"],
            num_class=len(self.labels),
            device=self._device,
        )
        self.adapter_model.load_state_dict(
            torch.load(model_path, map_location=self._device)
        )
        self.adapter_model.to(self._device)
        logger.info("Adapter Model Loaded")

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,
        input_data: pd.DataFrame,
        params: pd.DataFrame,
    ) -> pd.DataFrame:
        """Perform inference on the input data.

        :param context: MLflow context containing artifacts that the model can use for inference
        :type context: mlflow.pyfunc.PythonModelContext
        :param input_data: Input images and text for feature embeddings, please note that params are initialized in
            the file: medimageinsight_mlflow_model/MLmodel from the model's signature
            default values are: params: '[{"name": "image_standardization_jpeg_compression_ratio", "type": "integer",
            "default": 75, "shape": null}, {"name": "image_standardization_image_size", "type":"integer",
            "default": 512, "shape": null}]'
        :type input_data: Pandas DataFrame with a first column name ["image"] containing images where each
        row is an image in base64 String format or publicly accessible url format,
        and second column name ["text"] containing a string. The following cases are supported:
        - all rows in image column are populated with valid values and text column only contains empty string,
        - all rows in text column are populated with valid values and image column only contains empty string,
        - all rows in both columns are populated with valid values
        :return: Output of inferencing
        :rtype: Pandas DataFrame with columns "image_features" and/or "text_features"
        """
        from vision_utils import create_temp_file, process_image_pandas_series

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

            # Image Standardization
            if params["image_standardization_jpeg_compression_ratio"]:
                jpeg_compression_ratio = int(
                    params["image_standardization_jpeg_compression_ratio"]
                )
            else:
                jpeg_compression_ratio = 75

            if params["image_standardization_image_size"]:
                image_size = int(params["image_standardization_image_size"])
            else:
                image_size = 512

            image_features = self.run_inference_batch(
                image_path_list=image_path_list,
                jpeg_compression_ratio=jpeg_compression_ratio,
                image_size=image_size,
            )

        self.adapter_model.eval()
        results = []
        with torch.no_grad():
            for features in image_features.tolist():

                _, output = self.adapter_model(features)
                # Apply softmax to get probabilities
                probabilities = torch.softmax(output, dim=1)
                probabilities = probabilities.cpu().detach().numpy()
                # Collect predictions
                categories_and_probabilities = [
                    {"label": label, "score": float(prob)}
                    for label, prob in zip(self.labels, probabilities[0])
                ]
                categories_and_probabilities = sorted(
                    categories_and_probabilities, key=lambda x: x["score"], reverse=True
                )

                results.append(categories_and_probabilities)

        return results

    def run_inference_batch(
        self,
        image_path_list: List,
        image_size: int,
        jpeg_compression_ratio: int,
    ) -> Tuple[torch.tensor]:
        """Perform inference on batch of input images.

        :type image_path_list: List[str]
        :param text_list: list of text strings for inferencing
        :type text_list: List[str]
        :return: image features and text features
        :rtype: Tuple where each value is either torch.tensor of size (#inputs, 512) or None
        """
        output_image_feature = []
        self._model.eval()
        if image_path_list:
            for img_path in image_path_list:
                # Standardize Images before MedImageInsight Preprocessing Stage as JPEG input
                img = (
                    Image.open(img_path).resize((image_size, image_size)).convert("RGB")
                )
                buffered = BytesIO()
                img.save(buffered, format="JPEG", quality=jpeg_compression_ratio)
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                image = Image.open(BytesIO(base64.b64decode(img_str))).convert("RGB")
                img = image

                # MedImageInsight Preprocessing
                inputs = self.preprocess(img).unsqueeze(0)
                inputs = inputs.to(self._device)

                # Generate Image Features
                with torch.no_grad():
                    tmp_image_feature = self._model.encode_image(inputs)
                    output_image_feature.append(tmp_image_feature)

            image_features = torch.stack(output_image_feature, dim=0)
        else:
            image_features = None

        return image_features

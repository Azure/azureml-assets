# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""MLFlow pyfunc wrapper for Virchow models."""

import timm
import json

import mlflow.pyfunc
import torch
import pandas as pd
import io
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked
from config import MLflowSchemaLiterals
import logging
logger = logging.getLogger("mlflow")  # Set log level to debugging
logger.setLevel(logging.DEBUG)


class VirchowModelWrapper(mlflow.pyfunc.PythonModel):
    """MLFlow pyfunc wrapper for Virchow models."""

    def load_context(self, context):
        """Load a MLflow model with pyfunc.load_model().

        :param context: MLflow context containing artifacts that the model can use for inference
        :type context: mlflow.pyfunc.PythonModelContext
        """
        config_path = context.artifacts["config_path"]
        checkpoint_path = context.artifacts["checkpoint_path"]
        # config = json.loads(config_path.read_text())
        with open(config_path) as f:
            config = json.load(f)
        self.model = timm.create_model(
            model_name="vit_huge_patch14_224",
            checkpoint_path=checkpoint_path,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU,
            pretrained_cfg=config["pretrained_cfg"],
            **config["model_args"]
        )
        self.model.eval()
        self.transforms = create_transform(
            **resolve_data_config(self.model.pretrained_cfg, model=self.model)
        )

    # def predict(self, image_input_path: str, params: dict = None):
    def predict(
            self,
            context: mlflow.pyfunc.PythonModelContext,
            input_data: pd.DataFrame,
            params: pd.DataFrame) -> pd.DataFrame:
        """Perform inference on the input data.

        :param context: MLflow context containing artifacts that the model can use for inference
        :type context: mlflow.pyfunc.PythonModelContext
        :param input_data: Pandas DataFrame with columns ["image"], ["prompt"] and ["direct_question"], where
                           the image is either a url or a base64 string, the prompt is the dialog so far between the
                           user and the model and the direct question is a prompt with a single question from the user.
        :type input_data: pd.DataFrame
        :return: Pandas dataframe with column ["response"] containing the model's response to the dialog so far.
        """
        from vision_utils import process_image
        pil_images = [
            Image.open(io.BytesIO(process_image(image)))
            for image in input_data[MLflowSchemaLiterals.INPUT_COLUMN_IMAGE]
        ]
        # image = input_data["image"]
        # pil_image = Image.open(io.BytesIO(process_image(pil_images[0])))
        pil_image = self.transforms(pil_images[0]).unsqueeze(0)  # size: 1 x 3 x 224 x 224

        device_type = params.get("device_type", "cuda")
        to_half_precision = params.get("to_half_precision", False)

        with torch.inference_mode(), torch.autocast(
            device_type=device_type, dtype=torch.float16
        ):
            output = self.model(pil_image)  # size: 1 x 257 x 1280

        class_token = output[:, 0]  # size: 1 x 1280
        # patch_tokens = output[:, 1:]  # size: 1 x 256 x 1280

        # use the class token only as the embedding
        # size: 1 x 1280
        embedding = class_token

        # the model output will be fp32 because the final operation is a LayerNorm that is ran in mixed precision
        # optionally, you can convert the embedding to fp16 for efficiency in downstream use
        if to_half_precision:
            embedding = embedding.to(torch.float16)

        df_result = pd.DataFrame()
        df_result['output'] = embedding.tolist()
        return df_result

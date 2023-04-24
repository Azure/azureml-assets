# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Mlflow PythonModel wrapper class that loads the Mlflow model, preprocess inputs and performs inference."""

import base64
import io
import re
import subprocess
import sys
import tempfile

import mlflow
import numpy as np
import pandas as pd
import requests
import torch
from PIL import Image

from config import Tasks, MMDetLiterals, MLFlowSchemaLiterals, ODLiterals


def _create_temp_file(request_body: bytes, parent_dir: str) -> str:
    """Create temporory file, save image and return path to the file.

    :param request_body: Image
    :type request_body: bytes
    :param parent_dir: directory name
    :type parent_dir: str
    :return: Path to the file
    :rtype: str
    """
    with tempfile.NamedTemporaryFile(dir=parent_dir, mode="wb", delete=False) as image_file_fp:
        # image_file_fp.write(request_body)
        img_path = image_file_fp.name + ".png"
        img = Image.open(io.BytesIO(request_body))
        img.save(img_path)
        return img_path


def _process_image(img: pd.Series) -> pd.Series:
    """Process the image input (image url or base64 string) and return bytes.

    :param img: pandas series with image in base64 string format or url
    :type img: pd.Series
    :return: decoded image in pandas series format.
    :rtype: Pandas Series
    """
    image = img[0]
    if _is_valid_url(image):
        image = requests.get(image).content
        return pd.Series(image)
    try:
        return pd.Series(base64.b64decode(image))
    except Exception:
        raise ValueError("Invalid image format")


def _is_valid_url(text: str) -> bool:
    """Check if text is url or base64 string.

    :param text: text to validate
    :type text: str
    :return: True if url else false
    :rtype: bool
    """
    regex = (
        "((http|https)://)(www.)?"
        + "[a-zA-Z0-9@:%._\\+~#?&//=]"
        + "{2,256}\\.[a-z]"
        + "{2,6}\\b([-a-zA-Z0-9@:%"
        + "._\\+~#?&//=]*)"
    )
    p = re.compile(regex)

    # If the string is empty
    # return false
    if str is None:
        return False

    # Return if the string
    # matched the ReGex
    if re.search(p, text):
        return True
    else:
        return False


class ImagesDetectionMLFlowModelWrapper(mlflow.pyfunc.PythonModel):
    """MLFlow model wrapper for AutoML for Images models."""

    def __init__(
        self,
        task_type: str,
    ) -> None:
        """Mlflow model wrapper for AutoML for Images models.

        :param task_type: Task type used in training.
        :type task_type: str
        """
        super().__init__()
        self._model = None
        self._inference_detector = None
        self._task_type = task_type

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        """Load a Mlflow model with pyfunc.load_model().

        :param context: Mlflow context containing artifacts that the model can use for inference
        :type context: mlflow.pyfunc.PythonModelContext
        """
        print("Inside load_context()")

        if self._task_type == Tasks.MM_OBJECT_DETECTION.value:
            # Install mmcv and mmdet using mim, with pip installation is not working
            subprocess.check_call([sys.executable, "-m", "mim", "install", "mmcv-full==1.7.1"])
            subprocess.check_call([sys.executable, "-m", "mim", "install", "mmdet==2.28.2"])
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "opencv-python-headless==4.7.0.72", "--force-reinstall"]
            )

            # importing mmdet/mmcv afte installing using mim
            from mmdet.apis import inference_detector, init_detector
            from mmcv import Config

            self._inference_detector = inference_detector
            try:
                model_config_path = context.artifacts[MMDetLiterals.CONFIG_PATH]
                model_weights_path = context.artifacts[MMDetLiterals.WEIGHTS_PATH]

                _map_location = "cuda" if torch.cuda.is_available() else "cpu"
                _config = Config.fromfile(model_config_path)
                self._model = init_detector(_config, model_weights_path, device=_map_location)

                print("Model loaded successfully")
            except Exception:
                print("Failed to load the the model.")
                raise
        else:
            raise ValueError(f"invalid task type {self._task_type}")

    def predict(self, context: mlflow.pyfunc.PythonModelContext, input_data: pd.DataFrame) -> pd.DataFrame:
        """Perform inference on the input data.

        :param context: Mlflow context containing artifacts that the model can use for inference
        :type context: mlflow.pyfunc.PythonModelContext
        :param input_data: Input images for prediction
        :type input_data: Pandas DataFrame with a first column name ["image"] of images where each
        image is in base64 String format.
        :return: Output of inferencing
        :rtype: Pandas DataFrame with columns ["boxes"] for object detection
        """
        # process the images in image column
        processed_images = input_data.loc[:, [MLFlowSchemaLiterals.INPUT_COLUMN_IMAGE]].apply(
            axis=1, func=_process_image
        )

        with tempfile.TemporaryDirectory() as tmp_output_dir:
            image_path_list = (
                processed_images.iloc[:, 0].map(lambda row: _create_temp_file(row, tmp_output_dir)).tolist()
            )
            results = self._inference_detector(imgs=image_path_list, model=self._model)

            predictions = []
            for result in results:
                bboxes = np.vstack(result)
                labels = [np.full(bbox.shape[0], i, dtype=np.int32) for i, bbox in enumerate(result)]
                labels = np.concatenate(labels)

                cur_image_preds = {ODLiterals.BOXES: []}
                for bbox, label in zip(bboxes, labels):
                    cur_image_preds[ODLiterals.BOXES].append(
                        {
                            ODLiterals.BOX: {
                                ODLiterals.TOP_X: str(bbox[0]),
                                ODLiterals.TOP_Y: str(bbox[1]),
                                ODLiterals.BOTTOM_X: str(bbox[2]),
                                ODLiterals.BOTTOM_Y: str(bbox[3]),
                            },
                            ODLiterals.LABEL: self._model.CLASSES[label],
                            ODLiterals.SCORE: str(bbox[4]),
                        }
                    )
                predictions.append(cur_image_preds)
        return pd.DataFrame(predictions)

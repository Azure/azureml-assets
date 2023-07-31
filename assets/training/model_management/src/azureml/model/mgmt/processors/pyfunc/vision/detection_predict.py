# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""MLflow PythonModel wrapper class that loads the MLflow model, preprocess inputs and performs inference."""

import base64
import io
import re
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Tuple

import mlflow
import numpy as np
import pandas as pd
import requests
import torch
from PIL import Image
from config import Tasks, MMDetLiterals, MLflowSchemaLiterals, ODLiterals, ISLiterals


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
        img_path = image_file_fp.name + ".png"
        img = Image.open(io.BytesIO(request_body))
        img.save(img_path)
        return img_path


def _process_image(img: pd.Series) -> pd.Series:
    """Process input image.

    If input image is in bytes format, return it as it is.
    If input image is in base64 string format, decode it to bytes.
    If input image is in url format, download it and return bytes.
    https://github.com/mlflow/mlflow/blob/master/examples/flower_classifier/image_pyfunc.py

    :param img: pandas series with image in base64 string format or url or bytes.
    :type img: pd.Series
    :return: decoded image in pandas series format.
    :rtype: Pandas Series
    """
    image = img[0]
    if isinstance(image, bytes):
        return img
    elif isinstance(image, str):
        if _is_valid_url(image):
            image = requests.get(image).content
            return pd.Series(image)
        else:
            try:
                return pd.Series(base64.b64decode(image))
            except ValueError:
                raise ValueError(
                    "The provided image string cannot be decoded. Expected format is Base64 or URL string."
                )
    else:
        raise ValueError(
            f"Image received in {type(image)} format which is not supported."
            "Expected format is bytes, base64 string or url string."
        )


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


def _normalize_polygon(polygon: List[np.ndarray], image_size: Tuple[int, int]) -> List[np.ndarray]:
    """Normalize polygon coordinates.

    :param polygon: List of un-normalized polygons. Each points in polygon is a list of x, y coordinates.
    :rtype: List[np.ndarray]
    :param image_size: Image size
    :type image_size: Tuple[int, int]
    :return: List of normalized polygons.
    :rtype: List[np.ndarray]. List of arrays having x0, y0, x1, y1, ... polygon coordinates.
    """
    normalized_polygon = []
    for points in polygon:
        normalized_points = []
        for x, y in points:
            x = float(x) / image_size[0]
            y = float(y) / image_size[1]
            normalized_points.extend([x, y])
        normalized_polygon.append(normalized_points)
    return normalized_polygon


class ImagesDetectionMLflowModelWrapper(mlflow.pyfunc.PythonModel):
    """MLflow model wrapper for Images Detection models."""

    def __init__(
        self,
        task_type: str,
    ) -> None:
        """Convert AutoML Images models to MLflow.

        :param task_type: Task type used in training.
        :type task_type: str
        """
        super().__init__()
        self._model = None
        self._inference_detector = None
        self._task_type = task_type

    def _post_process_model_results(self, batch_results: List) -> List[Dict[str, Any]]:
        """Convert model results to OD or IS output format.
        :param batch_results: List of model results for images in batch.
        :type batch_results: List
        :return: List of predictions for images in batch.
        If task type is OD, each prediction is a dict of bbox, labels and classes.
        If task type is IS, each prediction is a dict of bbox, labels, classes and polygons.
        :rtype: List[Dict[str, Any]]
        """
        predictions = []
        for result in batch_results:
            image_height, image_width = result.ori_shape
            pred_instances = result.pred_instances
            bboxes = pred_instances.bboxes.numpy()
            labels = pred_instances.labels.numpy()
            scores = pred_instances.scores.numpy()
            masks = pred_instances.masks.numpy() if self._task_type == Tasks.MM_INSTANCE_SEGMENTATION.value else None

            cur_image_preds = {ODLiterals.BOXES: []}
            for i in range(len(labels)):
                box = {
                    ODLiterals.BOX: {
                        ODLiterals.TOP_X: float(bboxes[i][0]) / image_width,
                        ODLiterals.TOP_Y: float(bboxes[i][1]) / image_height,
                        ODLiterals.BOTTOM_X: float(bboxes[i][2]) / image_width,
                        ODLiterals.BOTTOM_Y: float(bboxes[i][3]) / image_height,
                    },
                    ODLiterals.LABEL: self.classes[labels[i]],
                    ODLiterals.SCORE: float(scores[i]),
                }
                if masks is not None:
                    from mmdet.structures.mask import bitmap_to_polygon

                    polygon, _ = bitmap_to_polygon(masks[i])
                    box[ISLiterals.POLYGON] = _normalize_polygon(polygon, (image_width, image_height))
                cur_image_preds[ODLiterals.BOXES].append(box)
            predictions.append(cur_image_preds)
        return predictions

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        """Load a MLflow model with pyfunc.load_model().

        :param context: MLflow context containing artifacts that the model can use for inference
        :type context: mlflow.pyfunc.PythonModelContext
        """
        print("Inside load_context()")

        if self._task_type in [Tasks.MM_OBJECT_DETECTION.value, Tasks.MM_INSTANCE_SEGMENTATION.value]:
            # Install mmcv and mmdet using mim, with pip installation is not working
            subprocess.check_call([sys.executable, "-m", "mim", "install", "mmcv==2.0.1"])
            subprocess.check_call([sys.executable, "-m", "mim", "install", "mmdet==3.1.0"])
            # mmdet installs opencv-python but it results in error while importing libGL.so.1. So, we
            # need to re-install headless version of opencv-python.
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "opencv-python-headless==4.7.0.72", "--force-reinstall"]
            )

            # importing mmdet/mmcv afte installing using mim
            from mmdet.apis import inference_detector, init_detector
            from mmengine.config import Config

            self._inference_detector = inference_detector
            try:
                model_config_path = context.artifacts[MMDetLiterals.CONFIG_PATH]
                model_weights_path = context.artifacts[MMDetLiterals.WEIGHTS_PATH]

                _map_location = "cuda" if torch.cuda.is_available() else "cpu"
                _config = Config.fromfile(model_config_path)
                self._model = init_detector(_config, model_weights_path, device=_map_location)
                self.classes = self._model.dataset_meta[MMDetLiterals.CLASSES]

                print("Model loaded successfully")
            except Exception:
                print("Failed to load the the model.")
                raise
        else:
            raise ValueError(f"invalid task type {self._task_type}")

    def predict(self, context: mlflow.pyfunc.PythonModelContext, input_data: pd.DataFrame) -> pd.DataFrame:
        """Perform inference on the input data.

        :param context: MLflow context containing artifacts that the model can use for inference
        :type context: mlflow.pyfunc.PythonModelContext
        :param input_data: Input images for prediction
        :type input_data: Pandas DataFrame with a first column name ["image"] of images where each
        image is in base64 String format.
        :return: Output of inferencing
        :rtype: Pandas DataFrame with columns ["boxes"] for object detection
        """
        # process the images in image column
        processed_images = input_data.loc[:, [MLflowSchemaLiterals.INPUT_COLUMN_IMAGE]].apply(
            axis=1, func=_process_image
        )

        with tempfile.TemporaryDirectory() as tmp_output_dir:
            image_path_list = (
                processed_images.iloc[:, 0].map(lambda row: _create_temp_file(row, tmp_output_dir)).tolist()
            )

            results = self._inference_detector(imgs=image_path_list, model=self._model)

            predictions = self._post_process_model_results(results)
            return pd.DataFrame(predictions)

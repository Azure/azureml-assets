# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""MLflow PythonModel wrapper class that loads the MLflow model, preprocess inputs and performs inference."""

import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Tuple

import mlflow
import numpy as np
import pandas as pd
import torch
from config import Tasks, MMDetLiterals, MLflowSchemaLiterals, ODLiterals, ISLiterals

try:
    # Use try/except since vision_utils is added as part of model export and not available when initializing
    # model wrapper for save_model().
    from vision_utils import create_temp_file, process_image_pandas_series
except ImportError:
    pass


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


def _remove_invalid_segments(polygon: List[List[float]]) -> List[List[float]]:
    """Remove invalid segments. Segment is valid if it contains more than or equal to 6 co-ordinates (triangle).

    :param polygon: List of polygon.
    :return: List of polygon.
    """
    return [segment for segment in polygon if len(segment) >= 6]


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
            bboxes = pred_instances.bboxes.cpu().numpy()
            labels = pred_instances.labels.cpu().numpy()
            scores = pred_instances.scores.cpu().numpy()
            masks = pred_instances.masks.cpu().numpy() if self._task_type == Tasks.MM_INSTANCE_SEGMENTATION.value else None

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
            axis=1, func=process_image_pandas_series
        )

        with tempfile.TemporaryDirectory() as tmp_output_dir:
            image_path_list = (
                processed_images.iloc[:, 0].map(lambda row: create_temp_file(row, tmp_output_dir)[0]).tolist()
            )

            results = self._inference_detector(imgs=image_path_list, model=self._model)

            predictions = self._post_process_model_results(results)
            return pd.DataFrame(predictions)

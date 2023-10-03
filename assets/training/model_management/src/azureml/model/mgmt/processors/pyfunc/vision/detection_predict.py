# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""MLflow PythonModel wrapper class that loads the MLflow model, preprocess inputs and performs inference."""

import subprocess
import sys
import tempfile
from typing import Dict, List, Tuple

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

    def _process_object_detection_results(
        self, results: List[np.ndarray], image_sizes: List[Tuple[int, int]]
    ) -> List[Dict[str, List]]:
        """Post process object detection model results.

        :param results: List of numpy arrays containing bounding boxes, labels and scores.
        :type results: List[np.ndarray]
        :param image_sizes: List of image sizes.
        :type image_sizes: List[Tuple[int, int]]
        :return: List of predictions having bounding boxes.
        :rtype: List[Dict[str, List]]
        """
        predictions = []
        for result, image_size in zip(results, image_sizes):
            bboxes = np.vstack(result)
            labels = [np.full(bbox.shape[0], i, dtype=np.int32) for i, bbox in enumerate(result)]
            labels = np.concatenate(labels)

            cur_image_preds = {ODLiterals.BOXES: []}
            for bbox, label in zip(bboxes, labels):
                cur_image_preds[ODLiterals.BOXES].append(
                    {
                        ODLiterals.BOX: {
                            ODLiterals.TOP_X: float(bbox[0]) / image_size[0],
                            ODLiterals.TOP_Y: float(bbox[1]) / image_size[1],
                            ODLiterals.BOTTOM_X: float(bbox[2]) / image_size[0],
                            ODLiterals.BOTTOM_Y: float(bbox[3]) / image_size[1],
                        },
                        ODLiterals.LABEL: self._model.CLASSES[label],
                        ODLiterals.SCORE: float(bbox[4]),
                    }
                )
            predictions.append(cur_image_preds)
        return predictions

    def _process_instance_segmentation_results(
        self, batch_predictions: List[Tuple[List[np.ndarray], List[np.ndarray]]], image_sizes: List[Tuple[int, int]]
    ) -> List[Dict[str, List]]:
        """Post process instance segmentation model results.

        :param batch_predictions: List of tuples containing bounding boxes and masks.
        :type batch_predictions: List[Tuple[List[np.ndarray], List[np.ndarray]]]
        :param image_sizes: List of image sizes.
        :type image_sizes: List[Tuple[int, int]]
        :return: List of predictions having bounding boxes and masks.
        :rtype: List[Dict[str, List]]
        """
        from mmcv import concat_list
        from mmdet.core.mask.structures import bitmap_to_polygon

        predictions = []
        for (predicted_bbox, predicted_mask), image_size in zip(batch_predictions, image_sizes):
            if isinstance(predicted_mask, tuple):
                predicted_mask = predicted_mask[0]  # ms rcnn
            bboxes = np.vstack(predicted_bbox)
            labels = [np.full(bbox.shape[0], i, dtype=np.int32) for i, bbox in enumerate(predicted_bbox)]
            labels = np.concatenate(labels)

            masks = concat_list(predicted_mask)  # Concatenate a list of list into a single list.
            if isinstance(masks[0], torch.Tensor):
                masks = torch.stack(masks, dim=0)
            else:
                masks = torch.as_tensor(np.stack(masks, axis=0))

            cur_image_preds = {ISLiterals.BOXES: []}
            for bbox, label, mask in zip(bboxes, labels, masks):
                polygon, _ = bitmap_to_polygon(mask)
                polygon = _remove_invalid_segments(polygon)
                if len(polygon) > 0:
                    cur_image_preds[ISLiterals.BOXES].append(
                        {
                            ISLiterals.BOX: {
                                ISLiterals.TOP_X: float(bbox[0]) / image_size[0],
                                ISLiterals.TOP_Y: float(bbox[1]) / image_size[1],
                                ISLiterals.BOTTOM_X: float(bbox[2]) / image_size[0],
                                ISLiterals.BOTTOM_Y: float(bbox[3]) / image_size[1],
                            },
                            ISLiterals.LABEL: self._model.CLASSES[label],
                            ISLiterals.SCORE: float(bbox[4]),
                            ISLiterals.POLYGON: _normalize_polygon(polygon, image_size),
                        }
                    )
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
            subprocess.check_call([sys.executable, "-m", "mim", "install", "mmcv-full==1.7.1"])
            subprocess.check_call([sys.executable, "-m", "mim", "install", "mmdet==2.28.2"])
            # mmdet installs opencv-python but it results in error while importing libGL.so.1. So, we
            # need to re-install headless version of opencv-python.
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
            image_path_list = []
            # Save image sizes to use it for normalization
            image_original_size_list = []
            for _, image in processed_images.iterrows():
                img_path, img = create_temp_file(image[0], tmp_output_dir)
                image_path_list.append(img_path)
                image_original_size_list.append(img.size)

            results = self._inference_detector(imgs=image_path_list, model=self._model)

            if self._task_type == Tasks.MM_OBJECT_DETECTION.value:
                predictions = self._process_object_detection_results(results, image_original_size_list)
            elif self._task_type == Tasks.MM_INSTANCE_SEGMENTATION.value:
                predictions = self._process_instance_segmentation_results(results, image_original_size_list)
            return pd.DataFrame(predictions)

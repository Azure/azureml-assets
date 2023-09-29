# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""MLflow PythonModel wrapper class that loads the MLflow model, preprocess inputs and performs inference."""

import base64
import io
import re
import subprocess
import sys
import tempfile
from typing import Dict, List, Tuple

import mlflow
import numpy as np
import pandas as pd
import requests
import torch
from PIL import Image
from config import Tasks, MMDetLiterals, MLflowSchemaLiterals, MOTLiterals

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


def _process_video(vid: pd.Series) -> str:
    """If input video is in url format, return the video url.
       This function called for each row in the input data, i.e one video a time.

    :param vid: pandas series with valid video url.
    :type vid: pd.Series
    :return: video link str.
    :rtype: str
    """
    video = vid[0]
    if isinstance(video, str):
        if _is_valid_url(video):
            return video
    raise ValueError("Video received is not in valid format. Expected format is url string.")


class VideosTrackingMLflowModelWrapper(mlflow.pyfunc.PythonModel):
    """MLFlow model wrapper for AutoML for Images models."""

    def __init__(
        self,
        task_type: str,
    ) -> None:
        """This method is called when the python model wrapper is initialized.

        :param task_type: Task type used in training.
        :type task_type: str
        """
        super().__init__()
        self._model = None
        self._task_type = task_type
        self._inference_detector = None
        self.video_reader = None

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        """This method is called when loading a Mlflow model with pyfunc.load_model().

        :param context: Mlflow context containing artifacts that the model can use for inference.
        :type context: mlflow.pyfunc.PythonModelContext
        """
        print("Inside load_context()")

        if self._task_type in ["video-multi-object-tracking"]:
            """
            Install mmtrack, mmcv and mmdet using mim, with pip installation is not working
            1. for mmtrack, one of its dependency is mmcv 1.6.2, which will trigger cuda related issues.
                to mitigate, we use no dependency install for mmtrack, and put other dependencies in pip requirement
            2. for opencv, the default installed by mmcv is opencv-python. however, it's installing unwanted UI,
                which causes problems for stability. thus we force reinstall opencv-python-headless.
            3. for numpy, we are reinstalling numpy to older version to be compatible to opencv. more info:
                https://stackoverflow.com/questions/20518632/importerror-numpy-core-multiarray-failed-to-import
            """
            # subprocess.check_call([sys.executable, "-m", "mim", "install", "mmcv-full==1.7.1"])
            # subprocess.check_call([sys.executable, "-m", "mim", "install", "mmdet==2.28.2"])
            # subprocess.check_call([sys.executable, "-m", "mim", "install", "--no-deps", "mmtrack==0.14.0"])
            # subprocess.check_call([sys.executable, "-m", "pip", "install",
            #                        "opencv-python-headless==4.7.0.72", "--force-reinstall"])
            # subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy==1.19.3", "--force-reinstall"])

            # importing mmdet/mmcv after installing using mim
            from mmtrack.apis import init_model, inference_mot
            import mmcv
            from mmcv import Config, VideoReader
            self._inference_detector = inference_mot
            self.video_reader = mmcv.VideoReader

            try:
                model_config_path = context.artifacts[MMDetLiterals.CONFIG_PATH]
                model_weights_path = context.artifacts[MMDetLiterals.WEIGHTS_PATH]

                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA is not available. GPU needed for MOT inference!")
                self._config = Config.fromfile(model_config_path)
                self._model = init_model(self._config)

                print("Model loaded successfully")
            except Exception:
                print("Failed to load the the model.")
                raise

        else:
            raise ValueError(f"invalid task type {self._task_type}."
                             f"Supported tasks: video-multi-object-tracking")

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
        processed_videos = input_data.loc[:, [MLflowSchemaLiterals.INPUT_COLUMN_VIDEO]].apply(
            axis=1, func=_process_video
        )
        all_predictions = []
        for _, video_url in processed_videos.iteritems():
            video_frames = self.video_reader(video_url)
            for j, img in enumerate(video_frames):
                result = self._inference_detector(self._model, img, frame_id=j)
                result = self._parse_mot_output(result, j, video_url)
                all_predictions.append(result)

        return pd.DataFrame(all_predictions)

    def _parse_mot_output(self, result, frame_id, video_url) -> List[Dict]:
        """Parse the output of inference_mot() to a list of dictionaries.
        param result: output of inference_mot()
        type result: tuple
        param frame_id: frame id of the current frame
        type frame_id: int
        param video_url: url of the video
        type video_url: str
        return: dictionary of current image predictions
        """
        print(result.keys())
        det_bboxes = result[MOTLiterals.DET_BBOXES]
        track_bboxes = result[MOTLiterals.TRACK_BBOXES]

        curimage_preds = {MOTLiterals.DET_BBOXES: [], MOTLiterals.TRACK_BBOXES: [],
                          MOTLiterals.FRAME_ID: frame_id, MOTLiterals.VIDEO_URL: video_url}
        for label, bboxes in enumerate(det_bboxes):
            for bbox in bboxes:
                curimage_preds[MOTLiterals.DET_BBOXES].append({
                    MOTLiterals.BOX: {
                        MOTLiterals.TOP_X: float(bbox[0]),
                        MOTLiterals.TOP_Y: float(bbox[1]),
                        MOTLiterals.BOTTOM_X: float(bbox[2]),
                        MOTLiterals.BOTTOM_Y: float(bbox[3]),
                    },
                    MOTLiterals.LABEL: label,
                    MOTLiterals.SCORE: float(bbox[4]),
                })
        for tlabel, tbboxes in enumerate(track_bboxes):
            for tbbox in tbboxes:
                curimage_preds[MOTLiterals.TRACK_BBOXES].append({
                    MOTLiterals.BOX: {
                        MOTLiterals.INSTANCE_ID: int(tbbox[0]),
                        MOTLiterals.TOP_X: float(tbbox[1]),
                        MOTLiterals.TOP_Y: float(tbbox[2]),
                        MOTLiterals.BOTTOM_X: float(tbbox[3]),
                        MOTLiterals.BOTTOM_Y: float(tbbox[4]),
                    },
                    MOTLiterals.LABEL: tlabel,
                    MOTLiterals.SCORE: float(tbbox[5]),
                })
        return curimage_preds

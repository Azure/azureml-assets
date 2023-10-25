# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""MLflow PythonModel wrapper class that loads the MLflow model, preprocess inputs and performs inference."""

import subprocess
import sys
from typing import Dict, List

import mlflow
import pandas as pd
import torch
from config import MMDetLiterals, MLflowSchemaLiterals, MOTLiterals, Tasks
try:
    # Use try/except since vision_utils is added as part of model export and not available when initializing
    # model wrapper for save_model().
    from vision_utils import _is_valid_url
except ImportError:
    pass


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
    """MLFlow model wrapper for AutoML Video Object Tracking models."""

    def __init__(
        self,
        task_type: str,
    ) -> None:
        """Convert AutoML video models to MLflow.

        :param task_type: Task type used in training.
        :type task_type: str
        """
        super().__init__()
        self._model = None
        self._task_type = task_type
        self._inference_detector = None
        self._video_reader = None

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        """Load a MLflow model with pyfunc.load_model().

        :param context: Mlflow context containing artifacts that the model can use for inference.
        :type context: mlflow.pyfunc.PythonModelContext
        """
        print("Inside load_context()")

        if self._task_type in [Tasks.MM_MULTI_OBJECT_TRACKING.value]:
            """
            Install mmtrack, mmcv and mmdet using mim, with pip installation is not working
            1. for mmtrack, one of its dependency is mmcv 1.6.2, which will trigger cuda related issues.
                to mitigate, we use no dependency install for mmtrack, and put other dependencies in pip requirement
            2. for opencv, the default installed by mmcv is opencv-python. however, it's installing unwanted UI,
                which causes problems for stability. thus we force reinstall opencv-python-headless.
            3. for numpy, we are reinstalling numpy to older version to be compatible to opencv. more info:
                https://stackoverflow.com/questions/20518632/importerror-numpy-core-multiarray-failed-to-import
            """
            subprocess.check_call([sys.executable, "-m", "mim", "install", "mmcv-full==1.7.1"])
            subprocess.check_call([sys.executable, "-m", "mim", "install", "mmdet==2.28.2"])
            subprocess.check_call([sys.executable, "-m", "mim", "install", "--no-deps", "mmtrack==0.14.0"])
            subprocess.check_call([sys.executable, "-m", "pip", "install",
                                   "opencv-python-headless==4.7.0.72", "--force-reinstall"])
            subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy==1.19.3", "--force-reinstall"])

            # importing mmdet/mmcv after installing using mim
            from mmtrack.apis import init_model, inference_mot
            import mmcv
            self._inference_detector = inference_mot
            self._video_reader = mmcv.VideoReader

            try:
                model_config_path = context.artifacts[MMDetLiterals.CONFIG_PATH]
                model_weights_path = context.artifacts[MMDetLiterals.WEIGHTS_PATH]

                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA is not available. GPU needed for MOT inference!")
                self._config = mmcv.Config.fromfile(model_config_path)
                self._model = init_model(self._config, model_weights_path)

                print("Model loaded successfully")
            except Exception:
                print("Failed to load the the model.")
                raise

        else:
            raise ValueError(f"invalid task type {self._task_type}."
                             f"Supported tasks: {Tasks.MM_MULTI_OBJECT_TRACKING.value}")

    def predict(self, context: mlflow.pyfunc.PythonModelContext, input_data: pd.DataFrame) -> pd.DataFrame:
        """Perform inference on the input data.

        :param context: MLflow context containing artifacts that the model can use for inference
        :type context: mlflow.pyfunc.PythonModelContext
        :param input_data: Input video link for prediction
        :type input_data: Pandas DataFrame with a first column name ["video"] of videos where each
        video is a publicly accessible video link.
        :return: Output of inferencing
        :rtype: Pandas DataFrame with columns ["boxes"] for object detection
        """
        # process the images in image column
        processed_videos = input_data.loc[:, [MLflowSchemaLiterals.INPUT_COLUMN_VIDEO]].apply(
            axis=1, func=_process_video
        )
        all_predictions = []
        for _, video_url in processed_videos.iteritems():
            video_frames = self._video_reader(video_url)
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

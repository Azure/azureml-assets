# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Stable Diffusion prediction function."""

import torch
import base64
import io
import pandas as pd
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from pathlib import Path
from typing import List


def _load_pyfunc(path: Path):
    """Load stable diffusion mlflow flavour model.

    :param path: path to the mlflow stable diffusion model.
    :type path: Path
    """
    return StableDiffusionInference(path)


class StableDiffusionInference:
    """Stable diffusion inference class.

    :param model_path: Path to mlflow stable diffusion model 
    :type model_path: Path
    """

    def __init__(self, model_path: Path):
        """Init.

        :param model_path: Path to mlflow stable diffusion model 
        :type model_path: Path
        """
        if torch.cuda.is_available():  # correct?
            device = "cuda"
            pipe = StableDiffusionPipeline.from_pretrained(
                model_path, local_files_only=True, torch_dtype=torch.float16)
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
            self._model = pipe.to(device)
        else:
            self._model = StableDiffusionPipeline.from_pretrained(model_path)

    def predict(self, model_input: pd.DataFrame) -> List[str]:
        """Return a stable diffusion image for the given model input.

        :param model_input: Model input in panda dataframes
        :type model_input: pd.DataFrame
        :return: Image encoded as base64 encoded string
        :rtype: List[str]
        """
        results = []
        for row in model_input.itertuples():
            task = row.task
            result = self._model(task)
            img = result.images[0]
            buf = io.BytesIO()
            img.save(buf, format='JPEG')
            img_base64 = base64.encodebytes(buf.getbuffer().tobytes()).decode('utf-8')
            results.append({
                'image': img_base64
            })
        return results

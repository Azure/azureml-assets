# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Stable Diffusion prediction function."""

import torch
import io
import base64
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from typing import List


def _load_pyfunc(path):
    """Load pyfunc flavour."""
    return DiffusionPyFunc(path)


class DiffusionPyFunc:
    """StableDiffusion pipeline."""

    def __init__(self, model_path):
        """Init."""
        if torch.cuda.is_available():  # correct?
            device = "cuda"
            pipe = StableDiffusionPipeline.from_pretrained(model_path, local_files_only=True, torch_dtype=torch.float16)
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
            self._model = pipe.to(device)
        else:
            self._model = StableDiffusionPipeline.from_pretrained(model_path)

    def predict(self, model_input) -> List[str]:
        """Predict."""
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

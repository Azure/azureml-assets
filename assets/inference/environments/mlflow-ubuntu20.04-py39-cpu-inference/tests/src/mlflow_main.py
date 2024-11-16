"""For mlflow main tests."""
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import os
from datetime import timedelta
import pytest

from utils import run_score_path_azmlinfsrv

DEFAULT_MODEL_DIR_NAME = "/var/model_dir"


@pytest.mark.skip(reason="Need infrastructure to build image for testing")
def test_score_path_azmlinfsrv_mlflow_before_2_0(inference_image_name, resource_directory, score_script):
    """For testing the scoring request gives proper response for mflow toy model."""
    env_vars = {
        "AZUREML_MODEL_DIR": DEFAULT_MODEL_DIR_NAME,
        "MLFLOW_MODEL_FOLDER": "mlflow_model_folder",
        "AZUREML_EXTRA_CONDA_YAML_ABS_PATH": DEFAULT_MODEL_DIR_NAME + "/mlflow_model_folder/conda.yaml",
    }

    data_path = os.path.join("..", "resources", "mlflow", "sample_input.json")

    with open(data_path) as f:
        payload_data = json.load(f)

    req = run_score_path_azmlinfsrv(
        inference_image_name,
        resource_directory,
        env_vars,
        payload_data=payload_data,
        check_text=False,
        overwrite_azuremlapp=False,
        custom_payload=True,
        poll_timeout=timedelta(seconds=480),
    )

    print(req._content)
    assert req.status_code == 200
    assert json.loads(req._content)[0] == {"0": 0}

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
    """
    Test scoring request gives proper response for mflow toy model
    """
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


@pytest.mark.skip(reason="Need infrastructure to build image for testing")
def test_score_path_azmlinfsrv_mlflow_after_2_0(inference_image_name, resource_directory, score_script):
    """
    Test scoring request gives proper response for mflow toy model
    """
    env_vars = {
        "AZUREML_MODEL_DIR": DEFAULT_MODEL_DIR_NAME,
        "MLFLOW_MODEL_FOLDER": "mlflow_2_0_model_folder",
        "AZUREML_EXTRA_CONDA_YAML_ABS_PATH": DEFAULT_MODEL_DIR_NAME + "/mlflow_2_0_model_folder/conda.yaml",
    }

    data_path = os.path.join("..", "resources", "mlflow", "sample_2_0_input.txt")

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
    assert json.loads(req._content)[0]['a'] == 3.0


@pytest.mark.skip(reason="Need infrastructure to build image for testing")
def test_swagger_mlflow(inference_image_name, resource_directory, score_script):
    """
    Test scoring request gives proper response for mflow toy model
    """
    env_vars = {
        "AZUREML_MODEL_DIR": DEFAULT_MODEL_DIR_NAME,
        "MLFLOW_MODEL_FOLDER": "mlflow_model_folder",
        "AZUREML_EXTRA_CONDA_YAML_ABS_PATH": DEFAULT_MODEL_DIR_NAME + "/mlflow_model_folder/conda.yaml",
    }

    data_path = os.path.join("..", "resources", "mlflow", "sample_input.txt")

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
        swagger=True,
        poll_timeout=timedelta(seconds=600),
    )

    assert json.loads(req._content)["definitions"]["ServiceInput"]["example"]["input_data"]["columns"][0] == "a"


@pytest.mark.skip(reason="Need infrastructure to build image for testing")
def test_score_path_azmlinfsrv_mlflow_no_sig(inference_image_name, resource_directory, score_script):
    """
    Test scoring request gives proper response for mflow toy model
    """
    env_vars = {
        "AZUREML_MODEL_DIR": DEFAULT_MODEL_DIR_NAME,
        "MLFLOW_MODEL_FOLDER": "mlflow_model_folder_no_sig",
        "AZUREML_EXTRA_CONDA_YAML_ABS_PATH": DEFAULT_MODEL_DIR_NAME + "/mlflow_model_folder_no_sig/conda.yaml",
    }

    data_path = os.path.join("..", "resources", "mlflow", "sample_input.txt")

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

    req = run_score_path_azmlinfsrv(
        inference_image_name,
        resource_directory,
        env_vars,
        payload_data=payload_data,
        check_text=False,
        overwrite_azuremlapp=False,
        custom_payload=True,
        swagger=True,
        poll_timeout=timedelta(seconds=600),
    )

    assert json.loads(req._content)["definitions"]["ServiceInput"]["example"]["input_data"] == {}


@pytest.mark.skip(reason="Need infrastructure to build image for testing")
def test_score_path_azmlinfsrv_mlflow_named_tensor(inference_image_name, resource_directory, score_script):
    """
    Test scoring request gives proper response for mflow toy model with named tensor input
    """
    env_vars = {
        "AZUREML_MODEL_DIR": DEFAULT_MODEL_DIR_NAME,
        "MLFLOW_MODEL_FOLDER": "mlflow_tensor_spec_named",
        "AZUREML_EXTRA_CONDA_YAML_ABS_PATH": DEFAULT_MODEL_DIR_NAME + "/mlflow_tensor_spec_named/conda.yaml",
    }

    data_path = os.path.join("..", "resources", "mlflow", "named_tensor_input.json")

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
    assert json.loads(req._content)[0] == 0.1


@pytest.mark.skip(reason="Need infrastructure to build image for testing")
def test_score_path_azmlinfsrv_mlflow_unnamed_tensor(inference_image_name, resource_directory, score_script):
    """
    Test scoring request gives proper response for mflow toy model with unnamed tensor input
    """
    env_vars = {
        "AZUREML_MODEL_DIR": DEFAULT_MODEL_DIR_NAME,
        "MLFLOW_MODEL_FOLDER": "mlflow_tensor_spec_unnamed",
        "AZUREML_EXTRA_CONDA_YAML_ABS_PATH": DEFAULT_MODEL_DIR_NAME + "/mlflow_tensor_spec_unnamed/conda.yaml",
    }

    data_path = os.path.join("..", "resources", "mlflow", "unnamed_tensor_input.json")

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
    assert json.loads(req._content)[0] == 0.1


@pytest.mark.skip(reason="Need infrastructure to build image for testing")
def test_score_path_azmlinfsrv_mlflow_params(inference_image_name, resource_directory, score_script):
    """
    Test scoring request gives proper response for mlflow toy model with params in signature and passed in
    """
    env_vars = {
        "AZUREML_MODEL_DIR": DEFAULT_MODEL_DIR_NAME,
        "MLFLOW_MODEL_FOLDER": "mlflow_model_params",
        "AZUREML_EXTRA_CONDA_YAML_ABS_PATH": DEFAULT_MODEL_DIR_NAME + "/mlflow_model_params/conda.yaml",
    }

    data_path = os.path.join("..", "resources", "mlflow", "sample_input_with_params.json")

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
    assert json.loads(req._content) == "my sentence 256"


@pytest.mark.skip(reason="Need infrastructure to build image for testing")
def test_score_path_azmlinfsrv_mlflow_params_not_passed(inference_image_name, resource_directory, score_script):
    """
    Test scoring request gives proper response for mlflow toy model with params in signature but not passed
    """
    env_vars = {
        "AZUREML_MODEL_DIR": DEFAULT_MODEL_DIR_NAME,
        "MLFLOW_MODEL_FOLDER": "mlflow_model_params",
        "AZUREML_EXTRA_CONDA_YAML_ABS_PATH": DEFAULT_MODEL_DIR_NAME + "/mlflow_model_params/conda.yaml",
    }

    data_path = os.path.join("..", "resources", "mlflow", "sample_input_with_params_not_passed.json")

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
    assert json.loads(req._content) == "my sentence 512"


@pytest.mark.skip(reason="Need infrastructure to build image for testing")
def test_score_path_azmlinfsrv_no_params(inference_image_name, resource_directory, score_script):
    """
    Test scoring request gives proper response for mflow toy model with unnamed tensor input
    """
    env_vars = {
        "AZUREML_MODEL_DIR": DEFAULT_MODEL_DIR_NAME,
        "MLFLOW_MODEL_FOLDER": "mlflow_no_params",
        "AZUREML_EXTRA_CONDA_YAML_ABS_PATH": DEFAULT_MODEL_DIR_NAME + "/mlflow_no_params/conda.yaml",
    }

    data_path = os.path.join("..", "resources", "mlflow", "no_params_df_input.json")

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
    output = [round(x, 4) for x in json.loads(req._content)]
    assert output == [10980.7116, 7763.8090]

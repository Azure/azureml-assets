"""
Tests running the pytorch_image_classifier/train.py script
on a randomly generated small dataset.
"""
import os
import sys
import tempfile
import pytest
from unittest.mock import patch

import numpy as np
from PIL import Image

from components.pytorch_pytorch_image_classifier import train
from components.pytorch_pytorch_image_classifier.model import MODEL_ARCH_LIST

# IMPORTANT: see conftest.py for fixtures


@pytest.fixture()
def random_image_in_folder_classes(temporary_dir):
    image_dataset_path = os.path.join(temporary_dir, "image_in_folders")
    os.makedirs(image_dataset_path, exist_ok=False)

    n_samples = 100
    n_classes = 4

    for i in range(n_samples):
        a = np.random.rand(300, 300, 3) * 255
        im_out = Image.fromarray(a.astype("uint8")).convert("RGB")

        class_dir = "class_{}".format(i % n_classes)

        image_path = os.path.join(
            image_dataset_path, class_dir, "random_image_{}.jpg".format(i)
        )
        os.makedirs(os.path.join(image_dataset_path, class_dir), exist_ok=True)
        im_out.save(image_path)

    return image_dataset_path

# IMPORTANT: we have to restrict the list of models for unit test
# because github actions runners have 7GB RAM only and will OOM
TEST_MODEL_ARCH_LIST = [
    "resnet18",
    "resnet34",
]

# we only care about patching those specific mlflow methods
@patch("mlflow.end_run") # we can have only 1 start/end per test session
@patch("mlflow.pytorch.log_model") # patched to test model name registration
@patch("mlflow.log_params") # patched to avoid conflict in parameters
@patch("mlflow.start_run") # we can have only 1 start/end per test session
@pytest.mark.parametrize("model_arch", TEST_MODEL_ARCH_LIST)
def test_components_pytorch_image_classifier(
    mlflow_start_run_mock,
    mlflow_log_params_mock,
    mlflow_pytorch_log_model_mock,
    mlflow_end_run_mock,
    model_arch,
    temporary_dir,
    random_image_in_folder_classes,
):
    """Tests src/components/pytorch_image_classifier/train.py"""
    model_dir = os.path.join(temporary_dir, "pytorch_image_classifier_model")

    # create test arguments for the script
    script_args = [
        "train.py",
        "--train_images",
        random_image_in_folder_classes,
        "--valid_images",
        random_image_in_folder_classes,  # using same data for train/valid
        "--batch_size",
        "16",
        "--num_workers",
        "0",  # single thread pre-fetching
        "--prefetch_factor",
        "2",  # will be discarded if num_workers=0
        "--pin_memory",
        "True",
        "--non_blocking",
        "False",
        "--model_arch",
        model_arch,
        "--model_arch_pretrained",
        "True",
        "--num_epochs",
        "1",
        "--model_output",
        model_dir,
        "--register_model_as",
        "foo",
        "--enable_profiling",
        "True",
    ]

    # replaces sys.argv with test arguments and run main
    with patch.object(sys, "argv", script_args):
        train.main()

    # those mlflow calls must be unique in the script
    mlflow_start_run_mock.assert_called_once()
    mlflow_end_run_mock.assert_called_once()

    # test all log_params calls
    for log_params_call in mlflow_log_params_mock.call_args_list:
        args, kwargs = log_params_call
        assert isinstance(args[0], dict) # call has only 1 argument, and it's a dict

    # test model registration
    log_model_calls = mlflow_pytorch_log_model_mock.call_args_list

    assert len(log_model_calls) == 1

    # unpack arguments
    args, kwargs = log_model_calls[0]
    assert "artifact_path" in kwargs
    assert kwargs["artifact_path"] == "final_model"
    assert "registered_model_name" in kwargs
    assert kwargs["registered_model_name"] == "foo"

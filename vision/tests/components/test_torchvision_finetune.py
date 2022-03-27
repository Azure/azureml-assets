"""
Tests running the torchvision_finetune/train.py script
on a randomly generated small dataset.
"""
import os
import sys
import tempfile
import pytest
from unittest.mock import patch

import numpy as np
from PIL import Image

from components.torchvision_finetune import train
from components.torchvision_finetune.model import MODEL_ARCH_LIST

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

@patch("mlflow.end_run")
@patch("mlflow.log_metric")
@patch("mlflow.set_tags")
@patch("mlflow.log_params")
@patch("mlflow.start_run")
@patch("mlflow.pytorch.log_model")
@pytest.mark.parametrize("model_arch", TEST_MODEL_ARCH_LIST)
def test_components_torchvision_finetune(
    mlflow_pytorch_log_model_mock,
    mlflow_start_run_mock,
    mlflow_log_params_mock,
    mlflow_set_tags_mock,
    mlflow_log_metric_mock,
    mlflow_end_run_mock,
    model_arch,
    temporary_dir,
    random_image_in_folder_classes,
):
    """Tests src/components/torchvision_finetune/train.py"""
    model_dir = os.path.join(temporary_dir, "torchvision_finetune_model")

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

    # test model registration
    log_model_calls = mlflow_pytorch_log_model_mock.call_args_list

    assert len(log_model_calls) == 1

    # unpack arguments
    args, kwargs = log_model_calls[0]
    assert "artifact_path" in kwargs
    assert kwargs["artifact_path"] == "final_model"
    assert "registered_model_name" in kwargs
    assert kwargs["registered_model_name"] == "foo"

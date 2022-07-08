"""
Tests running the pytorch_image_classifier/train.py script
on a randomly generated small dataset.
"""
import os
import sys
import pytest
from unittest.mock import patch

import numpy as np
from PIL import Image

from tensorflow_image_segmentation import train

# IMPORTANT: see conftest.py for fixtures


@pytest.fixture()
def random_images_and_masks(temporary_dir):
    image_dataset_path = os.path.join(temporary_dir, "segmentation", "images")
    os.makedirs(image_dataset_path, exist_ok=False)
    mask_dataset_path = os.path.join(temporary_dir, "segmentation", "masks")
    os.makedirs(mask_dataset_path, exist_ok=False)

    n_samples = 20
    n_classes = 3

    for i in range(n_samples):
        a = np.random.rand(300, 300, 3) * 255
        im_out = Image.fromarray(a.astype("uint8")).convert("RGB")

        a = np.random.randint(0, n_classes, size=(300, 300)) + 1  # pets collection starts index at 1
        msk_out = Image.fromarray(a.astype("uint8"))

        image_path = os.path.join(
            image_dataset_path, "random_image_{}.jpg".format(i)
        )
        im_out.save(image_path)

        mask_path = os.path.join(
            image_dataset_path, "random_image_{}.png".format(i)
        )
        msk_out.save(mask_path)

    return image_dataset_path


# IMPORTANT: we have to restrict the list of models for unit test
# because github actions runners have 7GB RAM only and will OOM
TEST_MODEL_ARCH_LIST = [
    "unet"
]


# we only care about patching those specific mlflow methods
@patch("mlflow.end_run")  # we can have only 1 start/end per test session
@patch("mlflow.log_params")  # patched to avoid conflict in parameters
@patch("mlflow.start_run")  # we can have only 1 start/end per test session
@pytest.mark.parametrize("model_arch", TEST_MODEL_ARCH_LIST)
def test_components_pytorch_image_classifier_single_node(
    mlflow_start_run_mock,
    mlflow_log_params_mock,
    mlflow_end_run_mock,
    model_arch,
    temporary_dir,
    random_images_and_masks,
):
    """Tests src/components/pytorch_image_classifier/train.py"""
    model_dir = os.path.join(temporary_dir, "tensorflow_image_segmentation_model")
    checkpoints_dir = os.path.join(
        temporary_dir, "tensorflow_image_segmentation_checkpoints"
    )

    # create test arguments for the script
    # fmt: off
    script_args = [
        "train.py",
        "--train_images", random_images_and_masks,
        "--images_filename_pattern", "(.*)\\.jpg",
        "--images_type", "jpg",
        "--train_masks", random_images_and_masks,
        "--masks_filename_pattern", "(.*)\\.png",
        "--test_images", random_images_and_masks,  # using same data for train/valid
        "--test_masks", random_images_and_masks,  # using same data for train/valid
        "--num_classes", "3",
        "--batch_size", "16",
        "--num_workers", "1",  # single thread pre-fetching
        "--prefetch_factor", "2",  # will be discarded if num_workers=0
        "--model_arch", model_arch,
        "--model_input_size", "160",
        "--num_epochs", "2",
        "--model_output", model_dir,
        "--checkpoints", checkpoints_dir,
    ]
    # fmt: on

    # replaces sys.argv with test arguments and run main
    with patch.object(sys, "argv", script_args):
        train.main()

    # those mlflow calls must be unique in the script
    mlflow_start_run_mock.assert_called_once()
    mlflow_end_run_mock.assert_called_once()

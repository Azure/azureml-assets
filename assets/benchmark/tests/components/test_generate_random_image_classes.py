# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Tests running the generate_random_image_classes/run.py script.
"""
import os
import sys
from unittest.mock import patch
import glob

from generate_random_image_classes import run

# IMPORTANT: see conftest.py for fixtures


def test_generate_random_image_classes(temporary_dir):
    """Tests src/components/generate_random_image_classes/run.py"""
    random_train_images = os.path.join(temporary_dir, "random_train_images")
    random_valid_images = os.path.join(temporary_dir, "random_valid_images")

    # create test arguments for the script
    script_args = [
        "run.py",
        "--output_train", random_train_images,
        "--output_valid", random_valid_images,
        "--width", "300",
        "--height", "300",
        "--classes", "4",
        "--train_samples", "400",
        "--valid_samples", "20",
    ]

    # replaces sys.argv with test arguments and run main
    with patch.object(sys, "argv", script_args):
        run.main()

    assert os.path.isdir(random_train_images)
    assert len(
        [file_path for file_path in glob.glob(random_train_images + "/**", recursive=True) if os.path.isfile(file_path)]
    ) == 400
    assert os.path.isdir(random_valid_images)
    assert len(
        [file_path for file_path in glob.glob(random_valid_images + "/**", recursive=True) if os.path.isfile(file_path)]
    ) == 20

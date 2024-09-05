# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Creates random images."""


import os

import numpy as np
from PIL import Image
import argparse


def generate(path, samples, classes, width, height):
    """Generate and save random images."""
    os.makedirs(path, exist_ok=True)

    for i in range(samples):
        a = np.random.rand(width, height, 3) * 255
        im_out = Image.fromarray(a.astype('uint8')).convert('RGB')

        class_dir = "class_{}".format(i % classes)

        image_path = os.path.join(
            path,
            class_dir,
            "random_image_{}.jpg".format(i)
        )
        os.makedirs(os.path.join(path, class_dir), exist_ok=True)
        im_out.save(image_path)


def main(cli_args: list = None) -> None:
    """Parse args and generate random images for training and validation sets."""
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "--output_train",
        type=str,
        required=True,
        help="Path to folder containing training images",
    )
    parser.add_argument(
        "--output_valid",
        type=str,
        required=True,
        help="path to folder containing validation images",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=300
    )
    parser.add_argument(
        "--height",
        type=int,
        default=300
    )
    parser.add_argument(
        "--classes",
        type=int,
        default=4
    )
    parser.add_argument(
        "--train_samples",
        type=int,
        default=500
    )
    parser.add_argument(
        "--valid_samples",
        type=int,
        default=100
    )

    # parse the arguments
    args = parser.parse_args(cli_args)

    # generate training set
    generate(args.output_train, args.train_samples, args.classes, args.width, args.height)

    # generate validation set
    generate(args.output_valid, args.valid_samples, args.classes, args.width, args.height)


if __name__ == "__main__":
    main()

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This script exports the CIFAR10 dataset to save it as actual files.

It writes them each under their respective class subdirectory.

To run:
> pip install tqdm torchvision numpy pillow
"""
import torchvision

import argparse
import os
from tqdm import tqdm


def run(args):
    """Download datasets from torchvision."""
    dataset_module = getattr(torchvision.datasets, args.dataset)

    print("*** Downloading datasets locally")

    train_download_dir = os.path.join(args.download_dir, args.dataset, "train")
    os.makedirs(train_download_dir, exist_ok=True)
    trainset = dataset_module(
        root=train_download_dir, train=True, download=True, transform=None
    )

    test_download_dir = os.path.join(args.download_dir, args.dataset, "train")
    os.makedirs(test_download_dir, exist_ok=True)
    testset = dataset_module(
        root=test_download_dir, train=False, download=True, transform=None
    )

    print(f"*** Saving training set of {args.dataset} as {args.export_type}")

    train_export_dir = os.path.join(args.output_dir, args.dataset, "train")
    os.makedirs(train_export_dir, exist_ok=True)

    image_index = 0
    for entry, target in tqdm(trainset):
        # one directory per class
        class_path = os.path.join(train_export_dir, str(target))
        os.makedirs(class_path, exist_ok=True)

        # save as image
        entry.save(os.path.join(class_path, f"{image_index}.{args.export_type}"))

        image_index += 1

    print(f"*** (saved {image_index})")

    print(f"*** Saving testing set of {args.dataset} as {args.export_type}")

    test_export_dir = os.path.join(args.output_dir, args.dataset, "train")
    os.makedirs(test_export_dir, exist_ok=True)

    image_index = 0
    for entry, target in tqdm(testset):
        # one directory per class
        class_path = os.path.join(test_export_dir, str(target))
        os.makedirs(class_path, exist_ok=True)

        # save as image
        entry.save(os.path.join(class_path, f"{image_index}.{args.export_type}"))

        image_index += 1

    print(f"saved {image_index}")


def main(cli_args=None):
    """Parse CLI args and call run."""
    parser = argparse.ArgumentParser(__doc__)

    parser.add_argument(
        "--output_dir", required=True, type=str, help="where to store dataset as JPG"
    )
    parser.add_argument(
        "--export_type", required=False, type=str, choices=["jpg", "png"], default="jpg"
    )
    parser.add_argument("--dataset", required=True, choices=["CIFAR10", "CIFAR100", "MNIST"])
    parser.add_argument(
        "--download_dir",
        required=False,
        type=str,
        default="/tmp/torchvision_download",
        help="where to download torchvision datasets",
    )

    # actually parses arguments now
    args = parser.parse_args(cli_args)  # if cli_args, uses sys.argv

    run(args)


if __name__ == "__main__":
    main()

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Pin images and packages and write results."""


import argparse
from pathlib import Path

import azureml.assets.environment as environment
from azureml.assets.util import logger


def transform_file(input_file: Path, output_file: Path = None):
    """Transform file."""
    # Read file
    with open(input_file) as f:
        contents = f.read()

    # Pin images and packages
    contents = environment.pin_images(contents)
    contents = environment.pin_packages(contents)

    # Write to stdout or output_file
    if output_file == "-":
        logger.print(contents)
    else:
        if output_file is None:
            output_file = input_file
        with open(output_file, "w") as f:
            f.write(contents)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=Path,
                        help="File containing images/packages to pin to latest versions", required=True)
    parser.add_argument("-o", "--output", type=Path,
                        help="File to which output will be written. Defaults to the input file.")
    args = parser.parse_args()

    output = args.output or args.input
    transform_file(args.input, output)

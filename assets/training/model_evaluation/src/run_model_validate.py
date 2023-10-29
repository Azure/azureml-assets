# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Validate Model Evaluation Pipeline Parameters."""

from utils import ArgumentParser


def run():
    """Entry function of model validation script."""
    parser = ArgumentParser()
    parser.add_argument("--validation-info", required=True, help="Model source")

    args, _ = parser.parse_known_args()

    print("Validation info: ", args.validation_info)
    with open(args.validation_info, "w") as f:
        f.write("Validation info")


if __name__ == "__main__":
    run()

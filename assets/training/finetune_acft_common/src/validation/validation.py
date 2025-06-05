# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""script to update validation info."""


def main():
    """Script which runs as part of validation component to update output."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-info", required=True, help="Model source ")

    args = parser.parse_args()

    print("Validation info: ", args.validation_info)
    with open(args.validation_info, "w") as f:
        f.write("Validation Completed")


if __name__ == "__main__":
    main()

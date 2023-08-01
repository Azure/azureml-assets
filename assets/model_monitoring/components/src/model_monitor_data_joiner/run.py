# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Entry script for Model Monitor Data Joiner Component."""

import argparse


def run():
    """Data Joiner."""
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--left_input_data", type=str, required=True)
    parser.add_argument("--left_join_column", type=str, required=True)
    parser.add_argument("--right_input_data", type=str, required=True)
    parser.add_argument("--right_join_column", type=str, required=True)
    # args = parser.parse_args()

    print('Successfully executed data joiner component.')


if __name__ == "__main__":
    run()

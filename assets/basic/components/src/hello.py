# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Hello World script."""
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument("--message")
args = parser.parse_args()

print(args.message)

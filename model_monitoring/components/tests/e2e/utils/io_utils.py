# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""This file contains IO utility functions."""

import uuid
from ruamel.yaml import YAML


def load_from_yaml(path: str):
    """Load a yaml file into a dict."""
    with open(path, "r") as stream:
        return YAML().load(stream)


def write_to_yaml(path: str, contents: dict):
    """Write a dict to a yaml."""
    with open(path, "w+", encoding="utf8") as outfile:
        yaml = YAML()
        yaml.default_flow_style = False
        yaml.dump(contents, outfile)


def print_header(msg: str):
    """Log a message header."""
    print("------------------------------")
    print(msg)
    print("------------------------------")


def generate_random_filename(extension: str):
    """Generate a GUID-based random file name with the given file extension."""
    return f"{str(uuid.uuid4())}.{extension}"

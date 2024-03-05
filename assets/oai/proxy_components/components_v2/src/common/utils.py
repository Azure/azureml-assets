# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""data upload component."""

import json
import yaml


def save_yaml(content, filename):
    """Save yaml file with given content and filename."""
    with open(filename, encoding='utf-8', mode='w') as fh:
        yaml.dump(content, fh)


def load_yaml(filename):
    """Load yaml file and return as dictionary."""
    with open(filename, encoding='utf-8') as fh:
        file_dict = yaml.load(fh, Loader=yaml.FullLoader)
    return file_dict


def load_json(filename):
    """Load json file and return as dictionary."""
    with open(filename, encoding='utf-8') as file:
        json_data = json.load(file)
    return json_data


def save_json(data, file_path):
    """Save dictionary to json file."""
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

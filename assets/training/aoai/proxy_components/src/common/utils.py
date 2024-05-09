# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""data upload component."""

import json
import yaml
from typing import Optional
import os
from io import BytesIO
import jsonlines
from openai.types.fine_tuning import FineTuningJobEvent
from urllib.parse import urlparse, unquote

train_dataset_split_ratio = 0.8


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


def get_train_validation_filename(train_file_path: str, validation_file_path: Optional[str]) -> tuple[str, str]:
    """Get display name for files from path."""
    train_file_name = os.path.basename(train_file_path)
    if validation_file_path is None:
        validation_file_name = "validation_" + train_file_name
    else:
        validation_file_name = os.path.basename(validation_file_path)
    return train_file_name, validation_file_name


def get_dataset_length(file_path: str) -> int:
    """Get total number of examples in file."""
    with jsonlines.open(file_path, mode='r') as file_reader:
        data_len = sum(1 for _ in file_reader)
    return data_len


def split_data_in_train_and_validation(file_path: str, split_index: int) -> tuple[BytesIO, BytesIO]:
    """Split train data to generate validation data."""
    index = 0
    train_data = BytesIO()
    validation_data = BytesIO()

    with open(file_path, "rb") as data_reader:
        for line in data_reader:
            if index < split_index:
                train_data.write(line)
            else:
                validation_data.write(line)
            index += 1
    return train_data, validation_data


def get_train_validation_data(train_file_path: str, validation_file_path: Optional[str]) -> tuple[BytesIO, BytesIO]:
    """Get train and validation data from files."""
    if validation_file_path is not None:
        train_data = open(train_file_path, "rb")
        validation_data = open(validation_file_path, "rb")
        return train_data, validation_data

    train_data_length = get_dataset_length(train_file_path)
    split_index = int(train_data_length * train_dataset_split_ratio)

    return split_data_in_train_and_validation(train_file_path, split_index)


def list_event_messages_after_given_event(events_list: list[FineTuningJobEvent], last_event_message: str) -> list[str]:
    """List of events after last event recorded."""
    event_message_list = []
    for event in events_list:
        if last_event_message == event.message:
            break
        event_message_list.append(event.message)
    event_message_list.reverse()
    return event_message_list


def parse_file_name_from_uri(file_uri: str) -> str:
    file_path = urlparse(file_uri).path
    filename = file_path.split('/')[-1]
    filename = unquote(filename)  # Decode URL-encoded characters if present
    return filename


def create_payload_for_data_upload_rest_call(file_uri: str) -> dict[str, str]:
    file_name = parse_file_name_from_uri(file_uri)
    payload: dict[str, str] = {
        "purpose": "fine-tune",
        "filename": file_name,
        "content_url": file_uri
    }
    return payload


def parse_file_id_from_upload_response(response: str) -> str:
    resp_json = json.loads(response)
    return resp_json["id"]

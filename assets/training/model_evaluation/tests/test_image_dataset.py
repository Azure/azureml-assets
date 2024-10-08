# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Test image dataset implementations."""

import json
import os
import pytest
import sys
import tempfile

from unittest.mock import patch

from azureml.acft.common_components.image.runtime_common.common.dataset_helper import AmlDatasetHelper

MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "./src"))
sys.path.append(MODEL_DIR)
from constants import TASK  # noqa: E402
from image_dataset import get_image_dataset  # noqa: E402


DATASET_PER_TASK = {
    TASK.IMAGE_CLASSIFICATION: [
        {"image_url": "AmlDatastore://images/a/image1.jpg", "label": 0},
        {"image_url": "AmlDatastore://images/a/image2.jpg", "label": 1},
    ],
    TASK.IMAGE_OBJECT_DETECTION: [
        {
            "image_url": "AmlDatastore://images/b/image1.png",
            "label": [{"label": 0, "topX": 0.0, "topY": 0.0, "bottomX": 0.5, "bottomY": 0.5}],
        },
        {
            "image_url": "AmlDatastore://images/b/image2.png",
            "label": [{"label": 1, "topX": 0.5, "topY": 0.5, "bottomX": 1.0, "bottomY": 1.0}],
        },
    ],
    TASK.IMAGE_GENERATION: [
        {"image_url": "example.com/image1.png", "label": "an example"},
        {"image_url": "example.com/image2.png", "label": "another example"},
    ],
}
MLTABLE_CONTENTS_PER_TASK = {
    TASK.IMAGE_CLASSIFICATION: (
        "paths:\n"
        "  - file: {file_name}\n"
        "transformations:\n"
        "  - read_json_lines:\n"
        "        encoding: utf8\n"
        "        invalid_lines: error\n"
        "        include_path_column: false\n"
        "  - convert_column_types:\n"
        "      - columns: image_url\n"
        "        column_type: stream_info\n"
        "type: mltable\n"
    ),
    TASK.IMAGE_OBJECT_DETECTION: (
        "paths:\n"
        "  - file: {file_name}\n"
        "transformations:\n"
        "  - read_json_lines:\n"
        "        encoding: utf8\n"
        "        invalid_lines: error\n"
        "        include_path_column: false\n"
        "  - convert_column_types:\n"
        "      - columns: image_url\n"
        "        column_type: stream_info\n"
        "type: mltable\n"
    ),
    TASK.IMAGE_GENERATION: (
        "paths:\n"
        "- file: {file_name}\n"
        "transformations:\n"
        "- read_json_lines:\n"
        "    encoding: utf8\n"
        "    include_path_column: false\n"
        "    invalid_lines: error\n"
        "    partition_size: 20971520\n"
        "    path_column: Path\n"
        "- convert_column_types:\n"
        "  - column_type: stream_info\n"
        "    columns: image_url\n"
        "type: mltable\n"
    ),
}


class MockWorkspace:
    """Mock workspace."""

    def __init__(self, subscription_id, resource_group, workspace_name, location, workspace_id):
        self.subscription_id = subscription_id
        self.resource_group = resource_group
        self._workspace_name = workspace_name
        self.location = location
        self._workspace_id_internal = workspace_id
        self.name = workspace_name


class MockExperiment:
    """Mock experiment."""

    def __init__(self, workspace, id):
        self.workspace = workspace
        self.id = id


class MockRun:
    """Mock run."""

    def __init__(self, id):
        self.id = id


class MockRunContext:
    """Mock run context."""

    def __init__(self, experiment, run_id, parent_run_id):
        self.experiment = experiment
        self._run_id = run_id
        self.id = run_id
        self.parent = MockRun(parent_run_id)


def get_mock_run_context():
    """Make mock run context."""
    TEST_EXPERIMENT_ID = "22222222-2222-2222-2222-222222222222"
    TEST_REGION = "eastus"
    TEST_PARENT_RUN_ID = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
    TEST_RESOURCE_GROUP = "testrg"
    TEST_RUN_ID = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    TEST_SUBSCRIPTION_ID = "00000000-0000-0000-0000-000000000000"
    TEST_WORKSPACE_ID = "11111111-1111-1111-111111111111"
    TEST_WORKSPACE_NAME = "testws"

    ws = MockWorkspace(
        subscription_id=TEST_SUBSCRIPTION_ID,
        resource_group=TEST_RESOURCE_GROUP,
        workspace_name=TEST_WORKSPACE_NAME,
        location=TEST_REGION,
        workspace_id=TEST_WORKSPACE_ID,
    )
    experiment = MockExperiment(workspace=ws, id=TEST_EXPERIMENT_ID)
    return MockRunContext(experiment, run_id=TEST_RUN_ID, parent_run_id=TEST_PARENT_RUN_ID)


@pytest.mark.parametrize("task_type,input_column_names,label_column_name", [
    (TASK.IMAGE_CLASSIFICATION, ["image_url"], "label"),
    (TASK.IMAGE_OBJECT_DETECTION, ["image_url"], "label"),
    (TASK.IMAGE_GENERATION, ["prompt"], "label"),
])
def test_image_dataset(task_type, input_column_names, label_column_name):
    """Test image dataset on small example."""
    with tempfile.TemporaryDirectory() as directory_name:
        # Save the jsonl file.
        dataset = DATASET_PER_TASK[task_type]
        with open(os.path.join(directory_name, "dataset.jsonl"), "wt") as f:
            for r in dataset:
                f.write(json.dumps(r) + "\n")

        # Save the MLTable file.
        mltable_str = MLTABLE_CONTENTS_PER_TASK[task_type].format(file_name="dataset.jsonl")
        with open(os.path.join(directory_name, "MLTable"), "wt") as f:
            f.write(mltable_str)

        # Make blank image files for image classification and object detection tasks, to simulate downloading.
        if task_type in [TASK.IMAGE_CLASSIFICATION, TASK.IMAGE_OBJECT_DETECTION]:
            for r in dataset:
                image_file_name_tokens = r["image_url"].replace("AmlDatastore://", "").split("/")
                os.makedirs(os.path.join(directory_name, *image_file_name_tokens[:-1]), exist_ok=True)
                open(os.path.join(directory_name, *image_file_name_tokens), "wb").close()

        # Load the MLTable.
        with patch("azureml.core.Run.get_context", get_mock_run_context), \
                patch(
                    "azureml.acft.common_components.image.runtime_common.common.utils.download_or_mount_image_files"
                ), \
                patch.object(AmlDatasetHelper, "get_data_dir", return_value=directory_name):
            df = get_image_dataset(task_type, directory_name, input_column_names, label_column_name)

        # Compare the loaded dataset with the original.
        if task_type == TASK.IMAGE_GENERATION:
            loaded_dataset = [{k: row[k] for k in ["prompt", "label"]} for _, row in df.iterrows()]
            for r1, r2 in zip(
                sorted(dataset, key=lambda x: x["label"]), sorted(loaded_dataset, key=lambda x: x["prompt"])
            ):
                assert r2 == {"prompt": r1["label"], "label": r1["image_url"]}

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains unit tests for the GSQ component."""

import json
import os
import sys
import shutil
import pytest
import uuid
import pandas as pd
import pyarrow as pa
from generation_safety_quality.annotation_compute_histogram.run import (
    apply_annotation,
    SIMILARITY,
    TEST_CONNECTION,
    THRESHOLD_PARAMS,
    ALL_METRIC_NAMES,
    PROMPT,
    CORRELATION_ID,
    TRACE_ID,
    ROOT_SPAN)
from shared_utilities import io_utils
from shared_utilities.momo_exceptions import (
    DataNotFoundError, InvalidInputError)
from shared_utilities.constants import MDC_CHAT_HISTORY_COLUMN
import spark_mltable  # noqa, to enable spark.read.mltable
from unittest.mock import patch


QUESTION = 'question'
EVALUATION = 'evaluation'
GPT_4 = 'gpt-4'
CHAT_HISTORY = MDC_CHAT_HISTORY_COLUMN


@pytest.fixture(scope="module")
def gsq_preprocessor_test_setup():
    """Change working directory to root of the assets/model_monitoring_components."""
    original_work_dir = os.getcwd()
    momo_work_dir = os.path.abspath(f"{os.path.dirname(__file__)}/../..")
    # change working directory to root of the assets/model_monitoring_components
    os.chdir(momo_work_dir)
    python_path = sys.executable
    os.environ["PYSPARK_PYTHON"] = python_path
    print("PYSPARK_PYTHON", os.environ.get("PYSPARK_PYTHON", "NA"))
    module_path = os.path.join(os.getcwd(), "src")
    old_python_path = os.environ.get("PYTHONPATH", None)
    old_python_path = f"{old_python_path};" if old_python_path else ""
    os.environ["PYTHONPATH"] = f"{old_python_path}{module_path}"
    print("PYTHONPATH:", os.environ.get("PYTHONPATH", "NA"))

    yield
    # change working directory back to original
    os.chdir(original_work_dir)
    # reset python path back to original
    os.environ["PYTHONPATH"] = old_python_path


@pytest.mark.gsq_test
@pytest.mark.unit
class TestGSQHistogram:
    """Test class for GSQ histogram component and utilities."""

    @patch("shared_utilities.io_utils.StoreUrl")
    def test_gsq_apply_annotation(self, MockStoreUrl, code_zip_test_setup,
                                  gsq_preprocessor_test_setup):
        """Test apply_annotation method in GSQ component."""
        mock_store_url = MockStoreUrl.return_value
        mock_store_url.get_credential.return_value = "mock credential"

        metric_names = [name for name in ALL_METRIC_NAMES if SIMILARITY not in name]
        call_apply_annotation(",".join(metric_names))

    @patch("shared_utilities.io_utils.StoreUrl")
    def test_gsq_apply_annotation_all_valid(self, MockStoreUrl, code_zip_test_setup,
                                            gsq_preprocessor_test_setup):
        """Test passing low threshold so that there is no violation table."""
        mock_store_url = MockStoreUrl.return_value
        mock_store_url.get_credential.return_value = "mock credential"

        metric_names = [name for name in ALL_METRIC_NAMES if SIMILARITY not in name]
        threshold_args = {threshold: 1 for threshold in THRESHOLD_PARAMS}
        call_apply_annotation(",".join(metric_names), threshold_args=threshold_args)

    def test_gsq_invalid_metrics(self, code_zip_test_setup,
                                 gsq_preprocessor_test_setup):
        """Test passing invalid metrics."""
        metric_names = ['some_invalid_metric_name', 'another_invalid_metric_name']
        joined_metric_names = ",".join(metric_names)
        with pytest.raises(InvalidInputError):
            call_apply_annotation(joined_metric_names)

    @patch("shared_utilities.io_utils.StoreUrl")
    def test_gsq_production_data_missing_required_cols(self, MockStoreUrl, code_zip_test_setup,
                                                       gsq_preprocessor_test_setup):
        """Test passing production data missing required columns."""
        mock_store_url = MockStoreUrl.return_value
        mock_store_url.get_credential.return_value = "mock credential"

        metric_names = [name for name in ALL_METRIC_NAMES if SIMILARITY not in name]
        missing_prompt_col = "missing_question"
        exp_err = "production_dataset must have column: " + missing_prompt_col
        with pytest.raises(InvalidInputError, match=exp_err):
            call_apply_annotation(
                ",".join(metric_names), prompt_column_name=missing_prompt_col)
        missing_completion_col = "missing_answer"
        exp_err = "production_dataset must have column: " + missing_completion_col
        with pytest.raises(InvalidInputError, match=exp_err):
            call_apply_annotation(
                ",".join(metric_names), completion_column_name=missing_completion_col)

    @patch("shared_utilities.io_utils.StoreUrl")
    def test_gsq_with_same_column_name(self, MockStoreUrl, code_zip_test_setup,
                                       gsq_preprocessor_test_setup):
        """Test passing same column name as in file for prompt."""
        mock_store_url = MockStoreUrl.return_value
        mock_store_url.get_credential.return_value = "mock credential"

        metric_names = [name for name in ALL_METRIC_NAMES if SIMILARITY not in name]
        mltable_path = get_mltable_path()
        # make copy of directory
        test_folder = "test_output_gsq_with_same_column_name"
        mltable_path_copy = os.path.abspath(os.path.join(os.getcwd(), test_folder))
        shutil.copytree(mltable_path, mltable_path_copy, dirs_exist_ok=True)
        # modify the file data.jsonl in folder to have same column name as in file
        with open(os.path.join(mltable_path_copy, "data.jsonl"), "r", encoding="utf8") as file:
            data = file.read()
        data = data.replace(f'"{QUESTION}":', f'"{PROMPT}":')
        with open(os.path.join(mltable_path_copy, "data.jsonl"), "w", encoding="utf8") as file:
            file.write(data)
        call_apply_annotation(
            ",".join(metric_names), prompt_column_name=PROMPT, mltable_path=mltable_path_copy)
        # remove test folder
        shutil.rmtree(mltable_path_copy)

    @patch("shared_utilities.io_utils.StoreUrl")
    def test_gsq_with_added_prompt_column_name(self, MockStoreUrl, code_zip_test_setup,
                                               gsq_preprocessor_test_setup):
        """Test dataset with extra prompt column, same as in requested dataset."""
        mock_store_url = MockStoreUrl.return_value
        mock_store_url.get_credential.return_value = "mock credential"

        metric_names = [name for name in ALL_METRIC_NAMES if SIMILARITY not in name]
        mltable_path = get_mltable_path()
        # make copy of directory
        test_folder = "test_output_gsq_with_added_prompt_column_name"
        mltable_path_copy = os.path.abspath(os.path.join(os.getcwd(), test_folder))
        shutil.copytree(mltable_path, mltable_path_copy, dirs_exist_ok=True)
        # modify the file data.jsonl in folder to have same column name as in file
        json_data = read_json_data(mltable_path_copy)
        for row in json_data:
            row[PROMPT] = row[QUESTION]
        write_json_data(mltable_path_copy, json_data)
        call_apply_annotation(
            ",".join(metric_names), prompt_column_name=PROMPT, mltable_path=mltable_path_copy)
        # remove test folder
        shutil.rmtree(mltable_path_copy)

    @patch("shared_utilities.io_utils.StoreUrl")
    def test_gsq_with_chat_history_column_name(self, MockStoreUrl, code_zip_test_setup,
                                               gsq_preprocessor_test_setup):
        """Test dataset with extra chat history column."""
        mock_store_url = MockStoreUrl.return_value
        mock_store_url.get_credential.return_value = "mock credential"

        metric_names = [name for name in ALL_METRIC_NAMES if SIMILARITY not in name]
        mltable_path = get_chat_history_mltable_path()
        call_apply_annotation(
            ",".join(metric_names), completion_column_name="output",
            context_column_name=CHAT_HISTORY, mltable_path=mltable_path)

    @pytest.mark.skip("need more mock on the MockStoreUrl")
    def test_gsq_with_added_passthrough_columns(self, code_zip_test_setup,
                                                gsq_preprocessor_test_setup):
        """Test dataset with extra passthrough columns added."""
        metric_names = [name for name in ALL_METRIC_NAMES if SIMILARITY not in name]
        mltable_path = get_mltable_path()
        # make copy of directory
        test_folder = "test_output_gsq_with_added_passthrough_columns"
        mltable_path_copy = os.path.abspath(os.path.join(os.getcwd(), test_folder))
        shutil.copytree(mltable_path, mltable_path_copy, dirs_exist_ok=True)
        # modify the file data.jsonl in folder to have additional columns added
        json_data = read_json_data(mltable_path_copy)
        for row in json_data:
            row[CORRELATION_ID] = str(uuid.uuid4())
            row[TRACE_ID] = str(uuid.uuid4())
            row[ROOT_SPAN] = 'dummy rootspan'
        write_json_data(mltable_path_copy, json_data)
        call_apply_annotation(",".join(metric_names), mltable_path=mltable_path_copy)
        # assert that the passthrough columns are added to the output
        test_path = get_test_path(None, "test_output")
        eval_path = os.path.join(test_path, EVALUATION)
        evaluation_df = io_utils.try_read_mltable_in_spark_with_error(eval_path, EVALUATION)
        for col in [CORRELATION_ID, TRACE_ID, ROOT_SPAN]:
            assert col in evaluation_df.columns
        # remove test folder
        shutil.rmtree(mltable_path_copy)

    @patch("shared_utilities.io_utils.StoreUrl")
    def test_gsq_with_empty_dataset(self, MockStoreUrl, code_zip_test_setup, gsq_preprocessor_test_setup):
        """Test passing empty dataset."""
        mock_store_url = MockStoreUrl.return_value
        mock_store_url.get_credential.return_value = "mock credential"

        metric_names = [name for name in ALL_METRIC_NAMES if SIMILARITY not in name]
        empty_mltable_path = write_empty_production_data()
        err_msg = "No data is found for input 'production_dataset'"
        with pytest.raises(DataNotFoundError, match=err_msg):
            call_apply_annotation(
                ",".join(metric_names), mltable_path=empty_mltable_path)
        # remove test folder
        shutil.rmtree(empty_mltable_path)


def read_json_data(mltable_path):
    """Read the json data from the mltable path."""
    with open(os.path.join(mltable_path, "data.jsonl"), "r", encoding="utf8") as file:
        data = file.read()
        data = data.split("\n")
    json_data = [json.loads(line) for line in data]
    return json_data


def write_json_data(mltable_path, json_data):
    """Write the json data to the mltable path."""
    data = "\n".join([json.dumps(row) for row in json_data])
    with open(os.path.join(mltable_path, "data.jsonl"), "w", encoding="utf8") as file:
        file.write(data)


def create_ml_table_file_contents(pq_filename):
    """Create MLTable file contents."""
    return (
        "$schema: http://azureml/sdk-2-0/MLTable.json\n"
        "type: mltable\n"
        "paths:\n"
        " - file: ./{0}\n"
        "transformations:\n"
        " - read_parquet\n"
    ).format(pq_filename)


def create_ml_table_file(path, contents):
    """Create MLTable file."""
    with open(os.path.join(path, "MLTable"), "w") as f:
        f.write(contents)


def write_empty_production_data():
    """Write an empty input data frame."""
    df = pd.DataFrame(columns=['question', 'answer', 'context'],
                      dtype=str)
    mltable_path = os.path.join(os.getcwd(), "empty_production_data")
    os.makedirs(mltable_path, exist_ok=True)
    pq_filename = "empty_production_data.parquet"
    pq_file_path = os.path.join(mltable_path, pq_filename)
    SCHEMA = pa.schema([('question', pa.string()), ('answer', pa.string()), ('context', pa.string())])
    df.to_parquet(pq_file_path, index=False, schema=SCHEMA)
    mltable_file_contents = create_ml_table_file_contents(pq_filename)
    create_ml_table_file(mltable_path, mltable_file_contents)
    return mltable_path


def get_chat_history_mltable_path():
    """Get chat history mltable path."""
    test_file_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(
        test_file_dir, "..", "e2e", "resources",
        "mltable_chat_history_data_small")


def get_mltable_path():
    """Get mltable path."""
    test_file_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(
        test_file_dir, "..", "e2e", "resources",
        "mltable_groundedness_preprocessed_target_small")


def get_test_path(root_path, name):
    """Get path to test output folder."""
    if root_path:
        path = os.path.join(root_path, name)
    else:
        path = os.path.abspath(os.path.join(os.getcwd(), name))
    if (not os.path.exists(path)):
        os.mkdir(path)
    return path


def call_apply_annotation(metric_names, prompt_column_name=QUESTION,
                          completion_column_name="answer",
                          context_column_name="context",
                          mltable_path=None,
                          threshold_args=None):
    """Call apply_annotation method in GSQ component."""
    if mltable_path is None:
        mltable_path = get_mltable_path()
    test_path = get_test_path(None, "test_output")
    histogram_path = get_test_path(test_path, "histogram")
    sample_index_path = get_test_path(test_path, "samples_index")
    histogram_file_path = histogram_path
    samples_index_file_path = sample_index_path

    if threshold_args is None:
        threshold_args = {threshold: 5 for threshold in THRESHOLD_PARAMS}
    violations = {
        "groundedness": "groundedness_violations",
        "relevance": "relevance_violations",
        "fluency": "fluency_violations",
        "similarity": "similarity_violations",
        "coherence": "coherence_violations",
    }

    for k, v in violations.items():
        violations[k] = get_test_path(test_path, v)

    evaluation_path = get_test_path(test_path, EVALUATION)

    apply_annotation(
        metric_names=metric_names,
        production_dataset=mltable_path,
        histogram=histogram_file_path,
        model_deployment_name=GPT_4,
        workspace_connection_arm_id=TEST_CONNECTION,
        num_samples=1,
        sample_rate=float(1),
        request_args={},
        endpoint_args={},
        threshold_args=threshold_args,
        prompt_column_name=prompt_column_name,
        completion_column_name=completion_column_name,
        context_column_name=context_column_name,
        ground_truth_column_name=None,
        samples_index=samples_index_file_path,
        violations=violations,
        evaluation=evaluation_path
    )

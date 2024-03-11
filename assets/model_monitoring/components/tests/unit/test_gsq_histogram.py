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
    _check_and_format_azure_endpoint_url,
    apply_annotation,
    SIMILARITY,
    AZURE_OPENAI_API_DEPLOYMENT_URL_PATTERN,
    AZURE_ENDPOINT_DOMAIN_VALID_PATTERN_RE,
    GPT_4,
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
import spark_mltable  # noqa, to enable spark.read.mltable


QUESTION = 'question'
EVALUATION = 'evaluation'


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

    def test_gsq_invalid_deployment_url(self):
        """Test _check_and_format_azure_endpoint_url method in GSQ component."""
        url_pattern = AZURE_OPENAI_API_DEPLOYMENT_URL_PATTERN
        domain_pattern_re = AZURE_ENDPOINT_DOMAIN_VALID_PATTERN_RE
        version = "2022-12-01"
        model = "test_model"
        invalid_url = "https://invalidurl.com"
        with pytest.raises(InvalidInputError):
            _check_and_format_azure_endpoint_url(
                url_pattern, domain_pattern_re, invalid_url,
                version, model)
        # this was the url causing the error
        cog_url = "australiaeast.api.cognitive.microsoft.com"
        with pytest.raises(InvalidInputError):
            _check_and_format_azure_endpoint_url(
                url_pattern, domain_pattern_re, cog_url, version, model)

    def test_gsq_valid_deployment_url(self):
        """Test _check_and_format_azure_endpoint_url method in GSQ component."""
        url_pattern = AZURE_OPENAI_API_DEPLOYMENT_URL_PATTERN
        domain_pattern_re = AZURE_ENDPOINT_DOMAIN_VALID_PATTERN_RE
        version = "2022-12-01"
        model = "test_model"
        valid_url = "abc.openai.azure.com"
        formatted_url = _check_and_format_azure_endpoint_url(
            url_pattern, domain_pattern_re, valid_url, version, model)
        expected_format = f"https://{valid_url}/openai/deployments/{model}?api-version={version}"
        assert formatted_url == expected_format

    def test_gsq_apply_annotation(self, code_zip_test_setup,
                                  gsq_preprocessor_test_setup):
        """Test apply_annotation method in GSQ component."""
        metric_names = [name for name in ALL_METRIC_NAMES if SIMILARITY not in name]
        call_apply_annotation(",".join(metric_names))

    def test_gsq_invalid_metrics(self, code_zip_test_setup,
                                 gsq_preprocessor_test_setup):
        """Test passing invalid metrics."""
        metric_names = ['some_invalid_metric_name', 'another_invalid_metric_name']
        joined_metric_names = ",".join(metric_names)
        with pytest.raises(InvalidInputError):
            call_apply_annotation(joined_metric_names)

    def test_gsq_production_data_missing_required_cols(self, code_zip_test_setup,
                                                       gsq_preprocessor_test_setup):
        """Test passing production data missing required columns."""
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

    def test_gsq_with_same_column_name(self, code_zip_test_setup,
                                       gsq_preprocessor_test_setup):
        """Test passing same column name as in file for prompt."""
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

    def test_gsq_with_added_prompt_column_name(self, code_zip_test_setup,
                                               gsq_preprocessor_test_setup):
        """Test dataset with extra prompt column, same as in requested dataset."""
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
        test_path = get_test_path()
        eval_path = os.path.join(test_path, EVALUATION)
        evaluation_df = io_utils.try_read_mltable_in_spark_with_error(eval_path, EVALUATION)
        for col in [CORRELATION_ID, TRACE_ID, ROOT_SPAN]:
            assert col in evaluation_df.columns
        # remove test folder
        shutil.rmtree(mltable_path_copy)

    def test_gsq_with_empty_dataset(self, code_zip_test_setup, gsq_preprocessor_test_setup):
        """Test passing empty dataset."""
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


def get_mltable_path():
    """Get mltable path."""
    test_file_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(
        test_file_dir, "..", "e2e", "resources",
        "mltable_groundedness_preprocessed_target_small")


def get_test_path():
    """Get path to test output folder."""
    return os.path.abspath(os.path.join(os.getcwd(), "test_output"))


def call_apply_annotation(metric_names, prompt_column_name=QUESTION,
                          completion_column_name="answer",
                          context_column_name="context",
                          mltable_path=None):
    """Call apply_annotation method in GSQ component."""
    if mltable_path is None:
        mltable_path = get_mltable_path()
    test_path = get_test_path()
    fuse_prefix = "file://"
    histogram_path = fuse_prefix + os.path.join(test_path, "histogram")
    samples_index_path = fuse_prefix + os.path.join(test_path, "samples_index")
    threshold_args = {threshold: 5 for threshold in THRESHOLD_PARAMS}
    violations = {
        "groundedness": "groundedness_violations",
        "relevance": "relevance_violations",
        "fluency": "fluency_violations",
        "similarity": "similarity_violations",
        "coherence": "coherence_violations",
    }

    for k, v in violations.items():
        violations[k] = fuse_prefix + os.path.join(test_path, v)

    evaluation_path = fuse_prefix + os.path.join(test_path, EVALUATION)

    apply_annotation(
        metric_names=metric_names,
        production_dataset=mltable_path,
        histogram=histogram_path,
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
        samples_index=samples_index_path,
        violations=violations,
        evaluation=evaluation_path
    )

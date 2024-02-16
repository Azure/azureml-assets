# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains unit tests for the GSQ component."""

import json
import os
import sys
import shutil
import pytest
import zipfile
import time
import random
import string
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
    PROMPT)
from shared_utilities.momo_exceptions import InvalidInputError
import spark_mltable  # noqa, to enable spark.read.mltable
from spark_mltable import SPARK_ZIP_PATH


QUESTION = "question"


@pytest.fixture(scope="session")
def gsq_zip_test_setup():
    """Zip files in module_path to src.zip."""
    momo_work_dir = os.path.abspath(f"{os.path.dirname(__file__)}/../..")
    module_path = os.path.join(momo_work_dir, "src")
    # zip files in module_path to src.zip
    s = string.ascii_lowercase + string.digits
    rand_str = '_' + ''.join(random.sample(s, 5))
    time_str = time.strftime("%Y%m%d-%H%M%S") + rand_str
    zip_path = os.path.join(module_path, f"src_{time_str}.zip")

    zf = zipfile.ZipFile(zip_path, "w")
    for dirname, subdirs, files in os.walk(module_path):
        for filename in files:
            abs_filepath = os.path.join(dirname, filename)
            rel_filepath = os.path.relpath(abs_filepath, start=module_path)
            print("zipping file:", rel_filepath)
            zf.write(abs_filepath, arcname=rel_filepath)
    zf.close()
    # add files to zip folder
    os.environ[SPARK_ZIP_PATH] = zip_path
    print("zip path set in gsq_preprocessor_test_setup: ", zip_path)

    yield
    # remove zip file
    os.remove(zip_path)
    # remove zip path from environment
    os.environ.pop(SPARK_ZIP_PATH, None)


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

    def test_gsq_apply_annotation(self, gsq_zip_test_setup,
                                  gsq_preprocessor_test_setup):
        """Test apply_annotation method in GSQ component."""
        metric_names = [name for name in ALL_METRIC_NAMES if SIMILARITY not in name]
        call_apply_annotation(",".join(metric_names))

    def test_gsq_invalid_metrics(self, gsq_zip_test_setup,
                                 gsq_preprocessor_test_setup):
        """Test passing invalid metrics."""
        metric_names = ['some_invalid_metric_name', 'another_invalid_metric_name']
        joined_metric_names = ",".join(metric_names)
        with pytest.raises(InvalidInputError):
            call_apply_annotation(joined_metric_names)

    def test_gsq_production_data_missing_required_cols(self, gsq_zip_test_setup,
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

    def test_gsq_with_same_column_name(self, gsq_zip_test_setup,
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

    def test_gsq_with_added_prompt_column_name(self, gsq_zip_test_setup,
                                               gsq_preprocessor_test_setup):
        """Test dataset with extra prompt column, same as in requested dataset."""
        metric_names = [name for name in ALL_METRIC_NAMES if SIMILARITY not in name]
        mltable_path = get_mltable_path()
        # make copy of directory
        test_folder = "test_output_gsq_with_added_prompt_column_name"
        mltable_path_copy = os.path.abspath(os.path.join(os.getcwd(), test_folder))
        shutil.copytree(mltable_path, mltable_path_copy, dirs_exist_ok=True)
        # modify the file data.jsonl in folder to have same column name as in file
        with open(os.path.join(mltable_path_copy, "data.jsonl"), "r", encoding="utf8") as file:
            data = file.read()
            data = data.split("\n")
        json_data = [json.loads(line) for line in data]
        for row in json_data:
            row[PROMPT] = row[QUESTION]
        data = "\n".join([json.dumps(row) for row in json_data])
        with open(os.path.join(mltable_path_copy, "data.jsonl"), "w", encoding="utf8") as file:
            file.write(data)
        call_apply_annotation(
            ",".join(metric_names), prompt_column_name=PROMPT, mltable_path=mltable_path_copy)
        # remove test folder
        shutil.rmtree(mltable_path_copy)


def get_mltable_path():
    """Get mltable path."""
    test_file_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(
        test_file_dir, "..", "e2e", "resources",
        "mltable_groundedness_preprocessed_target_small")


def call_apply_annotation(metric_names, prompt_column_name=QUESTION,
                          completion_column_name="answer",
                          context_column_name="context",
                          mltable_path=None):
    """Call apply_annotation method in GSQ component."""
    if mltable_path is None:
        mltable_path = get_mltable_path()
    test_path = os.path.abspath(os.path.join(os.getcwd(), "test_output"))
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
        violations=violations
    )

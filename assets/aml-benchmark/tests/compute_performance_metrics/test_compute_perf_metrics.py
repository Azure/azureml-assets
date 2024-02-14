# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for Compute Performance Metrics Component."""

from typing import List, Dict, Any, Optional
import json
import os
from math import isclose
import time
import uuid

import numpy as np
import pandas as pd
import pytest
from azure.ai.ml.entities import Job
from azure.ai.ml import Input

from ..test_utils import (
    load_yaml_pipeline,
    get_mlclient,
    Constants,
    download_outputs,
    get_mlflow_logged_metrics,
    get_src_dir,
    assert_exception_mssg,
    run_command,
)

test_perf_data_latencies = np.array([500, 500, 265, 265, 260, 260, 230])
test_perf_data_batch_sizes = np.array([2, 2, 2, 2, 2, 2, 1])
test_perf_data_input_tokens = np.array([250, 225, 230, 245, 260, 275, 270])
test_perf_data_output_tokens = np.array([4, 12, 6, 4, 6, 3, 6])
test_perf_data_input_chars = np.array([1000, 1000, 900, 1100, 1100, 1200, 1200])
test_perf_data_output_chars = np.array([12, 51, 22, 4, 22, 5, 22])


def _verify_and_get_output_records(
    output_path: str,
    percentiles: str,
    has_input_token_counts: bool = False,
    has_output_token_counts: bool = False,
    has_input_char_counts: bool = False,
    has_output_char_counts: bool = False,
) -> Dict[str, Any]:
    """Verify the output and get output records.

    :param output_path: The output file path.
    :type output_path: str
    :param percentiles: A comma-separated string indicating which latency percentiles to calculate.
    :type percentiles: str
    :param has_input_token_counts: Whether or not input token information was provided.
    :type has_input_token_counts: bool, optional
    :param has_output_token_counts: Whether or not output token information was provided.
    :type has_output_token_counts: bool, optional
    :param has_input_char_counts: Whether or not input character information was provided.
    :type has_input_char_counts: bool, optional
    :param has_output_char_counts: Whether or not output character information was provided.
    :type has_output_char_counts: bool, optional
    :return: output records
    :rtype: Dict[str, Any]
    """
    # Read the output file
    with open(output_path, "r") as f:
        output_records = [json.loads(line) for line in f]
    output_row_count = len(output_records)

    # Check row count and extract the metrics dictionary
    assert output_row_count == 1
    output_records = output_records[0]

    # Check presence of average metrics
    num_metrics = 0
    assert "latency_avg" in output_records
    num_metrics += 1
    if has_input_token_counts:
        assert "total_input_tokens" in output_records
        assert "input_tokens_per_sec" in output_records
        assert "latency_per_input_token_avg" in output_records
        num_metrics += 3
    if has_output_token_counts:
        assert "total_output_tokens" in output_records
        assert "output_tokens_per_sec" in output_records
        assert "latency_per_output_token_avg" in output_records
        num_metrics += 3
    if has_input_token_counts and has_output_token_counts:
        assert "latency_per_input_output_token_avg" in output_records
        num_metrics += 1

    if has_input_char_counts:
        assert "latency_per_input_char_avg" in output_records
        num_metrics += 1
    if has_output_char_counts:
        assert "latency_per_output_char_avg" in output_records
        num_metrics += 1
    if has_input_char_counts and has_output_char_counts:
        assert "latency_per_input_output_char_avg" in output_records
        num_metrics += 1

    # Check presence of percentile metrics
    for percentile in [
        float(i.strip()) for i in percentiles.split(",") if i and not i.isspace()
    ]:
        assert "latency_p{0}".format(percentile) in output_records
        num_metrics += 1
        if has_input_token_counts:
            assert "latency_per_input_token_p{0}".format(percentile) in output_records
            num_metrics += 1
        if has_output_token_counts:
            assert "latency_per_output_token_p{0}".format(percentile) in output_records
            num_metrics += 1
        if has_input_token_counts and has_output_token_counts:
            assert (
                "latency_per_input_output_token_p{0}".format(percentile)
                in output_records
            )
            num_metrics += 1
        if has_input_char_counts:
            assert "latency_per_input_char_p{0}".format(percentile) in output_records
            num_metrics += 1
        if has_output_char_counts:
            assert "latency_per_output_char_p{0}".format(percentile) in output_records
            num_metrics += 1
        if has_input_char_counts and has_output_char_counts:
            assert (
                "latency_per_input_output_char_p{0}".format(percentile)
                in output_records
            )
            num_metrics += 1
    assert "requests_per_sec" in output_records
    num_metrics += 1

    # Check that no other metrics are present
    assert num_metrics == len(output_records)

    return output_records


def _verify_metrics(
    output_records: Dict[str, Any],
    percentiles: str,
    has_input_token_counts: bool = False,
    has_output_token_counts: bool = False,
    has_input_char_counts: bool = False,
    has_output_char_counts: bool = False,
) -> None:
    """Verify the values of the output metrics.

    :param output_records: The metrics in the output file.
    :type output_records: Dict[str, Any]
    :param percentiles: A comma-separated string indicating which latency percentiles to calculate.
    :type percentiles: str
    :param has_input_token_counts: Whether or not input token information was provided.
    :type has_input_token_counts: bool, optional
    :param has_output_token_counts: Whether or not output token information was provided.
    :type has_output_token_counts: bool, optional
    :param has_input_char_counts: Whether or not input character information was provided.
    :type has_input_char_counts: bool, optional
    :param has_output_char_counts: Whether or not output character information was provided.
    :type has_output_char_counts: bool, optional
    :return: None.
    :rtype: NoneType
    """
    # Calculate normalized latencies
    batch_norm_latency = test_perf_data_latencies / test_perf_data_batch_sizes

    latency_per_input_token = batch_norm_latency / test_perf_data_input_tokens
    latency_per_output_token = batch_norm_latency / test_perf_data_output_tokens
    latency_per_input_output_token = batch_norm_latency / (
        test_perf_data_input_tokens + test_perf_data_output_tokens
    )

    latency_per_input_char = batch_norm_latency / test_perf_data_input_chars
    latency_per_output_char = batch_norm_latency / test_perf_data_output_chars
    latency_per_input_output_char = batch_norm_latency / (
        test_perf_data_input_chars + test_perf_data_output_chars
    )

    # Check value of average metrics
    assert isclose(output_records["latency_avg"], np.average(batch_norm_latency))
    if has_input_token_counts:
        assert isclose(
            output_records["total_input_tokens"], np.sum(test_perf_data_input_tokens)
        )
        assert isclose(
            output_records["latency_per_input_token_avg"],
            np.average(latency_per_input_token),
        )
    if has_output_token_counts:
        assert isclose(
            output_records["total_output_tokens"], np.sum(test_perf_data_output_tokens)
        )
        assert isclose(
            output_records["latency_per_output_token_avg"],
            np.average(latency_per_output_token),
        )
    if has_input_token_counts and has_output_token_counts:
        assert isclose(
            output_records["latency_per_input_output_token_avg"],
            np.average(latency_per_input_output_token),
        )

    if has_input_char_counts:
        assert isclose(
            output_records["latency_per_input_char_avg"],
            np.average(latency_per_input_char),
        )
    if has_output_char_counts:
        assert isclose(
            output_records["latency_per_output_char_avg"],
            np.average(latency_per_output_char),
        )
    if has_input_char_counts and has_output_char_counts:
        assert isclose(
            output_records["latency_per_input_output_char_avg"],
            np.average(latency_per_input_output_char),
        )

    # Check value of percentile metrics
    for percentile in [
        float(i.strip()) for i in percentiles.split(",") if i and not i.isspace()
    ]:
        assert isclose(
            output_records["latency_p{0}".format(percentile)],
            np.percentile(batch_norm_latency, percentile),
        )
        if has_input_token_counts:
            assert isclose(
                output_records["latency_per_input_token_p{0}".format(percentile)],
                np.percentile(latency_per_input_token, percentile),
            )
        if has_output_token_counts:
            assert isclose(
                output_records["latency_per_output_token_p{0}".format(percentile)],
                np.percentile(latency_per_output_token, percentile),
            )
        if has_input_token_counts and has_output_token_counts:
            assert isclose(
                output_records[
                    "latency_per_input_output_token_p{0}".format(percentile)
                ],
                np.percentile(latency_per_input_output_token, percentile),
            )
        if has_input_char_counts:
            assert isclose(
                output_records["latency_per_input_char_p{0}".format(percentile)],
                np.percentile(latency_per_input_char, percentile),
            )
        if has_output_char_counts:
            assert isclose(
                output_records["latency_per_output_char_p{0}".format(percentile)],
                np.percentile(latency_per_output_char, percentile),
            )
        if has_input_char_counts and has_output_char_counts:
            assert isclose(
                output_records["latency_per_input_output_char_p{0}".format(percentile)],
                np.percentile(latency_per_input_output_char, percentile),
            )


class TestComputePerfMetricsComponent:
    """Tests for the compute performance metrics component."""

    EXP_NAME = "compute-perf-metrics-test"

    def test_compute_performance_metrics_component(
        self,
        temp_dir: str,
    ) -> None:
        """Compute Performance Metrics component test."""
        ml_client = get_mlclient()

        percentiles = "50,90,99"
        pipeline_job = self._get_pipeline_job(
            self.test_compute_performance_metrics_component.__name__,
            "batch_size",
            "start_time_iso",
            "end_time_iso",
            "input_token_count",
            "output_token_count",
            "input_char_count",
            "output_char_count",
            percentiles,
            temp_dir,
        )

        # submit the pipeline job
        pipeline_job = ml_client.create_or_update(
            pipeline_job, experiment_name=self.EXP_NAME
        )
        ml_client.jobs.stream(pipeline_job.name)
        print(pipeline_job)

        out_dir = os.path.join(temp_dir, "output")
        os.makedirs(out_dir, exist_ok=True)
        output_records = self._verify_and_get_output_records(
            pipeline_job,
            out_dir,
            percentiles,
            has_input_token_counts=True,
            has_output_token_counts=True,
            has_input_char_counts=True,
            has_output_char_counts=True,
        )

        _verify_metrics(
            output_records,
            percentiles,
            has_input_token_counts=True,
            has_output_token_counts=True,
            has_input_char_counts=True,
            has_output_char_counts=True,
        )

        self._verify_logged_metrics(
            pipeline_job.name,
            output_records,
            percentiles,
            has_input_token_counts=True,
            has_output_token_counts=True,
            has_input_char_counts=True,
            has_output_char_counts=True,
        )

    @pytest.mark.parametrize(
        "has_input_tokens, has_output_tokens, has_input_chars, has_output_chars",
        [
            (True, True, False, False),
            (False, False, True, True),
            (True, False, True, False),
        ],
    )
    def test_correct_metrics_when_omitting_columns(
        self,
        temp_dir: str,
        has_input_tokens: bool,
        has_output_tokens: bool,
        has_input_chars: bool,
        has_output_chars: bool,
    ) -> None:
        """Test that the correct metrics are emitted when omitting_columns."""
        ml_client = get_mlclient()

        percentiles = "50,90,99"
        pipeline_job = self._get_pipeline_job(
            self.test_correct_metrics_when_omitting_columns.__name__,
            "batch_size",
            "start_time_iso",
            "end_time_iso",
            "input_token_count" if has_input_tokens else None,
            "output_token_count" if has_output_tokens else None,
            "input_char_count" if has_input_chars else None,
            "output_char_count" if has_output_chars else None,
            percentiles,
            temp_dir,
        )

        # submit the pipeline job
        pipeline_job = ml_client.create_or_update(
            pipeline_job, experiment_name=self.EXP_NAME
        )
        ml_client.jobs.stream(pipeline_job.name)
        print(pipeline_job)

        out_dir = os.path.join(temp_dir, "output")
        os.makedirs(out_dir, exist_ok=True)
        output_records = self._verify_and_get_output_records(
            pipeline_job,
            out_dir,
            percentiles,
            has_input_token_counts=has_input_tokens,
            has_output_token_counts=has_output_tokens,
            has_input_char_counts=has_input_chars,
            has_output_char_counts=has_output_chars,
        )

        _verify_metrics(
            output_records,
            percentiles,
            has_input_token_counts=has_input_tokens,
            has_output_token_counts=has_output_tokens,
            has_input_char_counts=has_input_chars,
            has_output_char_counts=has_output_chars,
        )

        self._verify_logged_metrics(
            pipeline_job.name,
            output_records,
            percentiles,
            has_input_token_counts=has_input_tokens,
            has_output_token_counts=has_output_tokens,
            has_input_char_counts=has_input_chars,
            has_output_char_counts=has_output_chars,
        )

    @pytest.mark.parametrize("percentiles", ["10.5,24,30,99.999", "10", "", None])
    def test_correct_percentiles(
        self, temp_dir: str, percentiles: Optional[str]
    ) -> None:
        """Test that the correct percentile metrics are emitted when modifying the percentile value."""
        ml_client = get_mlclient()

        pipeline_job = self._get_pipeline_job(
            self.test_correct_percentiles.__name__,
            "batch_size",
            "start_time_iso",
            "end_time_iso",
            "input_token_count",
            "output_token_count",
            "input_char_count",
            "output_char_count",
            percentiles,
            temp_dir,
        )
        if percentiles is None:
            percentiles = "50,90,99"

        # submit the pipeline job
        pipeline_job = ml_client.create_or_update(
            pipeline_job, experiment_name=self.EXP_NAME
        )
        ml_client.jobs.stream(pipeline_job.name)
        print(pipeline_job)

        out_dir = os.path.join(temp_dir, "output")
        os.makedirs(out_dir, exist_ok=True)
        output_records = self._verify_and_get_output_records(
            pipeline_job,
            out_dir,
            percentiles,
            has_input_token_counts=True,
            has_output_token_counts=True,
            has_input_char_counts=True,
            has_output_char_counts=True,
        )

        _verify_metrics(
            output_records,
            percentiles,
            has_input_token_counts=True,
            has_output_token_counts=True,
            has_input_char_counts=True,
            has_output_char_counts=True,
        )

        self._verify_logged_metrics(
            pipeline_job.name,
            output_records,
            percentiles,
            has_input_token_counts=True,
            has_output_token_counts=True,
            has_input_char_counts=True,
            has_output_char_counts=True,
        )

    def _get_pipeline_job(
        self,
        display_name: str,
        batch_size_column_name: str,
        start_time_column_name: str,
        end_time_column_name: str,
        input_token_count_column_name: Optional[str] = None,
        output_token_count_column_name: Optional[str] = None,
        input_char_count_column_name: Optional[str] = None,
        output_char_count_column_name: Optional[str] = None,
        percentiles: Optional[str] = None,
        temp_dir: Optional[str] = None,
    ) -> Job:
        """Get the pipeline job.

        :param display_name: The display name for job.
        :type display_name: str
        :param batch_size_column_name: The name of the column that contains batch size information.
        :type batch_size_column_name: str
        :param start_time_column_name: The name of the column that contains start time information.
        :type start_time_column_name: str
        :param end_time_column_name: The name of the column that contains end time information.
        :type end_time_column_name: str
        :param input_token_count_column_name: The name of the column that contains input token information. If none
            is provided, the default in the yaml file will be used.
        :type input_token_count_column_name: Optional[str], optional
        :param output_token_count_column_name: The name of the column that contains output token information.
        :type output_token_count_column_name: Optional[str], optional
        :param input_char_count_column_name: The name of the column that contains input character information.
        :type input_char_count_column_name: Optional[str], optional
        :param output_char_count_column_name: The name of the column that contains output character information.
        :type output_char_count_column_name: Optional[str], optional
        :param percentiles: A comma-separated string indicating which latency percentiles to calculate. If None is
            provided, the default percentiles will be used; to calculate no percentiles, pass in the empty string.
        :type percentiles: Optional[str], optional
        :return: The pipeline job.
        :rtype: Job
        """
        pipeline_job = load_yaml_pipeline("compute_perf_metrics_pipeline.yaml")

        # avoid blob exists error when running pytest with multiple workers
        if temp_dir is not None:
            file_path = os.path.join(temp_dir, uuid.uuid4().hex + ".jsonl")
            with open(Constants.PERF_INPUT_FILE_PATH, "r") as f:
                with open(file_path, "w") as f2:
                    f2.write(f.read())

        # set the pipeline inputs
        pipeline_job.inputs.performance_data = Input(
            type="uri_file", path=file_path
        )
        pipeline_job.inputs.batch_size_column_name = batch_size_column_name
        pipeline_job.inputs.start_time_column_name = start_time_column_name
        pipeline_job.inputs.end_time_column_name = end_time_column_name

        pipeline_job.inputs.input_token_count_column_name = (
            input_token_count_column_name
        )
        pipeline_job.inputs.output_token_count_column_name = (
            output_token_count_column_name
        )
        pipeline_job.inputs.input_char_count_column_name = input_char_count_column_name
        pipeline_job.inputs.output_char_count_column_name = (
            output_char_count_column_name
        )
        if percentiles is not None:
            pipeline_job.inputs.percentiles = percentiles

        pipeline_job.display_name = display_name

        return pipeline_job

    def _verify_and_get_output_records(
        self,
        job: Job,
        output_dir: str,
        percentiles: str,
        has_input_token_counts: bool = False,
        has_output_token_counts: bool = False,
        has_input_char_counts: bool = False,
        has_output_char_counts: bool = False,
    ) -> Dict[str, Any]:
        """Verify the output and get output records.

        :param job: The job object.
        :type job: Job
        :param output_dir: The existing output directory to download the output in.
        :type output_dir: str
        :param percentiles: A comma-separated string indicating which latency percentiles to calculate.
        :type percentiles: str
        :param has_input_token_counts: Whether or not input token information was provided.
        :type has_input_token_counts: bool, optional
        :param has_output_token_counts: Whether or not output token information was provided.
        :type has_output_token_counts: bool, optional
        :param has_input_char_counts: Whether or not input character information was provided.
        :type has_input_char_counts: bool, optional
        :param has_output_char_counts: Whether or not output character information was provided.
        :type has_output_char_counts: bool, optional
        :return: output records
        :rtype: List[Dict[str, Any]]
        """
        output_name = job.outputs.performance_result.port_name
        download_outputs(
            job_name=job.name, output_name=output_name, download_path=output_dir
        )
        output_file_path = Constants.OUTPUT_FILE_PATH.format(
            output_dir=output_dir,
            output_name=output_name,
            output_file_name="performance_result.jsonl",  # taken from the pipeline's output path
        )
        return _verify_and_get_output_records(
            output_file_path,
            percentiles,
            has_input_token_counts=has_input_token_counts,
            has_output_token_counts=has_output_token_counts,
            has_input_char_counts=has_input_char_counts,
            has_output_char_counts=has_output_char_counts,
        )

    def _verify_logged_metrics(
        self,
        job_name: str,
        output_records: Dict[str, Any],
        percentiles: str,
        has_input_token_counts: bool = False,
        has_output_token_counts: bool = False,
        has_input_char_counts: bool = False,
        has_output_char_counts: bool = False,
    ) -> None:
        """Verify the logged metrics.

        :param job_name: The job name.
        :type job_name: str
        :param output_records: The output records from the component
        :type sampling_style: Dict[str, Any]
        :param percentiles: A comma-separated string indicating which latency percentiles to calculate.
        :type percentiles: str
        :param has_input_token_counts: Whether or not input token information was provided.
        :type has_input_token_counts: bool, optional
        :param has_output_token_counts: Whether or not output token information was provided.
        :type has_output_token_counts: bool, optional
        :param has_input_char_counts: Whether or not input character information was provided.
        :type has_input_char_counts: bool, optional
        :param has_output_char_counts: Whether or not output character information was provided.
        :type has_output_char_counts: bool, optional
        :return: None.
        :rtype: NoneType
        """
        logged_metrics = get_mlflow_logged_metrics(job_name, self.EXP_NAME)
        counter = 0
        while not logged_metrics:
            time.sleep(10)
            # It looks like sometimes the metrics is not flowed, and we need to wait a little bit.
            logged_metrics = get_mlflow_logged_metrics(job_name, self.EXP_NAME)
            print(f"logger metrics is:{logged_metrics} with counter {counter}")
            counter += 1
            if counter > 6:
                break

        # Verify the logged parameters
        assert logged_metrics["latency_avg"] == output_records["latency_avg"]
        if has_input_token_counts:
            assert (
                logged_metrics["total_input_tokens"]
                == output_records["total_input_tokens"]
            )
            assert (
                logged_metrics["latency_per_input_token_avg"]
                == output_records["latency_per_input_token_avg"]
            )
        if has_output_token_counts:
            assert (
                logged_metrics["total_output_tokens"]
                == output_records["total_output_tokens"]
            )
            assert (
                logged_metrics["latency_per_output_token_avg"]
                == output_records["latency_per_output_token_avg"]
            )
        if has_input_token_counts and has_output_token_counts:
            assert (
                logged_metrics["latency_per_input_output_token_avg"]
                == output_records["latency_per_input_output_token_avg"]
            )

        if has_input_char_counts:
            assert (
                logged_metrics["latency_per_input_char_avg"]
                == output_records["latency_per_input_char_avg"]
            )
        if has_output_char_counts:
            assert (
                logged_metrics["latency_per_output_char_avg"]
                == output_records["latency_per_output_char_avg"]
            )
        if has_input_char_counts and has_output_char_counts:
            assert (
                logged_metrics["latency_per_input_output_char_avg"]
                == output_records["latency_per_input_output_char_avg"]
            )

        # Check presence of various percentile metrics
        for percentile in [
            float(i.strip()) for i in percentiles.split(",") if i and not i.isspace()
        ]:
            assert (
                logged_metrics["latency_p{0}".format(percentile)]
                == output_records["latency_p{0}".format(percentile)]
            )
            if has_input_token_counts:
                assert (
                    logged_metrics["latency_per_input_token_p{0}".format(percentile)]
                    == output_records["latency_per_input_token_p{0}".format(percentile)]
                )
            if has_output_token_counts:
                assert (
                    logged_metrics["latency_per_output_token_p{0}".format(percentile)]
                    == output_records[
                        "latency_per_output_token_p{0}".format(percentile)
                    ]
                )
            if has_input_token_counts and has_output_token_counts:
                assert (
                    logged_metrics[
                        "latency_per_input_output_token_p{0}".format(percentile)
                    ]
                    == output_records[
                        "latency_per_input_output_token_p{0}".format(percentile)
                    ]
                )
            if has_input_char_counts:
                assert (
                    logged_metrics["latency_per_input_char_p{0}".format(percentile)]
                    == output_records["latency_per_input_char_p{0}".format(percentile)]
                )
            if has_output_char_counts:
                assert (
                    logged_metrics["latency_per_output_char_p{0}".format(percentile)]
                    == output_records["latency_per_output_char_p{0}".format(percentile)]
                )
            if has_input_char_counts and has_output_char_counts:
                assert (
                    logged_metrics[
                        "latency_per_input_output_char_p{0}".format(percentile)
                    ]
                    == output_records[
                        "latency_per_input_output_char_p{0}".format(percentile)
                    ]
                )


class TestComputePerfMetricsScript:
    """Tests for compute performance metrics script."""

    @pytest.mark.parametrize(
        "incorrect_column",
        [
            "batch_size",
            "start_time",
            "end_time",
            "input_tokens",
            "output_tokens",
            "input_characters",
            "output_characters",
        ],
    )
    def test_invalid_column_names(self, temp_dir: str, incorrect_column: str):
        """Test for invalid column names."""
        # Create test data
        test_data = [
            {
                "batch_size": 1,
                "start_time": "1970-01-01T00:00:00+00:00",
                "end_time": "1970-01-01T00:00:01+00:00",
                "input_tokens": 10,
                "output_tokens": 3,
                "input_characters": 42,
                "output_characters": 10,
            }
        ]

        # Create input file, column name list, and expected exception message
        input_file_path = self._create_input_file(
            temp_dir, file_name="temp_test.jsonl", data=test_data
        )
        cols = [
            "batch_size",
            "start_time",
            "end_time",
            "input_tokens",
            "output_tokens",
            "input_characters",
            "output_characters",
        ]
        if incorrect_column == "batch_size":
            incorrect_col_name = "batch_size_wrong"
            cols[0] = incorrect_col_name
            expected_exception_mssg = (
                f"'{incorrect_col_name}' was provided as the batch size column but no such column exists "
                "in the provided data. "
            )
        elif incorrect_column == "start_time":
            incorrect_col_name = "start_time_wrong"
            cols[1] = incorrect_col_name
            expected_exception_mssg = (
                f"'{incorrect_col_name}' was provided as the start time column but no such column exists "
                "in the provided data. "
            )
        elif incorrect_column == "end_time":
            incorrect_col_name = "end_time_wrong"
            cols[2] = incorrect_col_name
            expected_exception_mssg = (
                f"'{incorrect_col_name}' was provided as the end time column but no such column exists "
                "in the provided data. "
            )
        elif incorrect_column == "input_tokens":
            incorrect_col_name = "input_tokens_wrong"
            cols[3] = incorrect_col_name
            expected_exception_mssg = (
                f"'{incorrect_col_name}' was provided as the input token column but no such column exists "
                "in the provided data. "
            )
        elif incorrect_column == "output_tokens":
            incorrect_col_name = "output_tokens_wrong"
            cols[4] = incorrect_col_name
            expected_exception_mssg = (
                f"'{incorrect_col_name}' was provided as the output token column but no such column exists "
                "in the provided data. "
            )
        elif incorrect_column == "input_characters":
            incorrect_col_name = "input_characters_wrong"
            cols[5] = incorrect_col_name
            expected_exception_mssg = (
                f"'{incorrect_col_name}' was provided as the input character column but no such column exists "
                "in the provided data. "
            )
        elif incorrect_column == "output_characters":
            incorrect_col_name = "output_characters_wrong"
            cols[6] = incorrect_col_name
            expected_exception_mssg = (
                f"'{incorrect_col_name}' was provided as the output character column but no such column exists "
                "in the provided data. "
            )

        # Run the script and verify the exception
        try:
            self._run_perf_metrics_script(
                input_file_path,
                "50",
                cols[0],
                cols[1],
                cols[2],
                input_token_count_column_name=cols[3],
                output_token_count_column_name=cols[4],
                input_char_count_column_name=cols[5],
                output_char_count_column_name=cols[6],
            )
        except RuntimeError as e:
            exception_message = str(e)
            assert_exception_mssg(exception_message, expected_exception_mssg)

    @pytest.mark.parametrize(
        "percentiles", [["50:"], ["10", "101"], ["50.2", "99.5", "-1"]]
    )
    def test_invalid_percentiles(self, temp_dir: str, percentiles: List[str]):
        """Test for invalid percentiles."""
        # Create test data
        test_data = [
            {
                "start_time_iso": "1970-01-01T00:00:00+00:00",
                "end_time_iso": "1970-01-01T00:00:01+00:00",
                "batch_size": 1,
            }
        ]

        # Create input file and expected exception message
        input_file_path = self._create_input_file(
            temp_dir, file_name="temp_test.jsonl", data=test_data
        )
        expected_exception_mssg = None
        for percentile in percentiles:
            valid_percentile = True
            try:
                new_percentile = float(percentile)
                if new_percentile < 0 or new_percentile > 100:
                    valid_percentile = False
            except ValueError:
                valid_percentile = False

            if not valid_percentile:
                expected_exception_mssg = f"'{percentile}' was provided as a percentile but is not a valid percentile."
        assert expected_exception_mssg is not None

        # Run the script and verify the exception
        try:
            self._run_perf_metrics_script(
                input_file_path,
                ",".join(percentiles),
                "batch_size",
                "start_time_iso",
                "end_time_iso",
            )
        except RuntimeError as e:
            exception_message = str(e)
            assert_exception_mssg(exception_message, expected_exception_mssg)

    def _create_input_file(
        self, directory: str, file_name: str, data: List[Dict[str, Any]]
    ) -> str:
        """Create an input file.

        :param directory: The existing directory to create the file in.
        :type directory: str
        :param file_name: The file name with extension, either `.json` or `.jsonl`.
        :type file_name: str
        :param data: The data to put in the input file.
        :type file_name: str
        :return: The created input file path.
        :rtype: str
        """
        file_path = os.path.join(directory, file_name)
        file_content = pd.DataFrame(data)
        file_content.to_json(file_path, orient="records", lines=True)
        return file_path

    def _run_perf_metrics_script(
        self,
        dataset: str,
        percentiles: str,
        batch_size_column_name: str,
        start_time_column_name: str,
        end_time_column_name: str,
        input_token_count_column_name: Optional[str] = None,
        output_token_count_column_name: Optional[str] = None,
        input_char_count_column_name: Optional[str] = None,
        output_char_count_column_name: Optional[str] = None,
        output: str = "output.jsonl",
    ) -> None:
        """
        Run the compute performance metrics script with the given arguments.

        :param dataset: The input dataset.
        :type dataset: str
        :param percentiles: A comma-separated string indicating which latency percentiles to calculate.
        :type percentiles: str
        :param batch_size_column_name: The name of the column that contains batch size information.
        :type batch_size_column_name: str
        :param start_time_column_name: The name of the column that contains start time information.
        :type start_time_column_name: str
        :param end_time_column_name: The name of the column that contains end time information.
        :type end_time_column_name: str
        :param input_token_count_column_name: The name of the column that contains input token information.
        :type input_token_count_column_name: Optional[str], optional
        :param output_token_count_column_name: The name of the column that contains output token information.
        :type output_token_count_column_name: Optional[str], optional
        :param input_char_count_column_name: The name of the column that contains input character information.
        :type input_char_count_column_name: Optional[str], optional
        :param output_char_count_column_name: The name of the column that contains output character information.
        :type output_char_count_column_name: Optional[str], optional
        :param output: The file where the output information will be stored, defaults to "output.jsonl"
        :type output: str, optional
        :return: None
        :rtype: NoneType
        """
        src_dir = get_src_dir()
        args = [
            f"cd {src_dir} &&",
            "python -m aml_benchmark.compute_performance_metrics.main",
            "--performance_data",
            dataset,
            "--batch_size_column_name",
            batch_size_column_name,
            "--start_time_column_name",
            start_time_column_name,
            "--end_time_column_name",
            end_time_column_name,
            "--performance_result",
            output,
        ]
        if input_token_count_column_name is not None:
            args.extend(
                ["--input_token_count_column_name", input_token_count_column_name]
            )
        if output_token_count_column_name is not None:
            args.extend(
                ["--output_token_count_column_name", output_token_count_column_name]
            )
        if input_char_count_column_name is not None:
            args.extend(
                ["--input_char_count_column_name", input_char_count_column_name]
            )
        if output_char_count_column_name is not None:
            args.extend(
                ["--output_char_count_column_name", output_char_count_column_name]
            )
        if percentiles is not None:
            args.extend(["--percentiles", percentiles])

        run_command(" ".join(args))

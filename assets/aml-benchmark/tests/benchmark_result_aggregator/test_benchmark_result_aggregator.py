# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for Benchmarking Result Aggregator Component."""

import os
import json

from azure.ai.ml.entities import Job
import pytest

from ..test_utils import (
    load_yaml_pipeline,
    get_mlclient,
    download_outputs,
)


class TestBenchmarkResultAggregatorComponent:
    """Tests for dataset downloader component."""

    EXP_NAME = "benchmark-result-aggregator-test"

    @pytest.mark.parametrize(
        "pipeline_file_name, has_perf_step, has_quality_step",
        [
            ("benchmark_result_aggregator_pipeline.yaml", True, True),
            ("benchmark_result_aggregator_pipeline_no_quality.yaml", True, False),
            ("benchmark_result_aggregator_pipeline_no_perf.yaml", False, True),
        ],
    )
    def test_benchmark_result_aggregator_component(
        self,
        temp_dir: str,
        pipeline_file_name: str,
        has_perf_step: bool,
        has_quality_step: bool,
    ) -> None:
        """Benchmark result aggregator component test."""
        ml_client = get_mlclient()

        pipeline_job = self._get_pipeline_job(
            pipeline_file_name,
            self.test_benchmark_result_aggregator_component.__name__,
        )

        # submit the pipeline job
        pipeline_job = ml_client.create_or_update(
            pipeline_job, experiment_name=self.EXP_NAME
        )
        ml_client.jobs.stream(pipeline_job.name)
        print(pipeline_job)

        self._verify_output(pipeline_job, temp_dir, has_quality_step, has_perf_step)

    def _get_pipeline_job(
        self,
        pipeline_file_name: str,
        display_name: str,
    ) -> Job:
        """Get the pipeline job.

        :param pipeline_file_name: Name of the pipeline file to load.
        :param display_name: Display name for the job.
        :return: The pipeline job.
        """
        pipeline_job = load_yaml_pipeline(pipeline_file_name)
        pipeline_job.display_name = display_name
        return pipeline_job

    def _verify_output(
        self,
        job: Job,
        output_dir: str,
        has_quality_step: bool,
        has_perf_step: bool,
    ) -> None:
        """Verify the output.

        :param job: The job object.
        :param output_dir: The existing output directory to download the output in.
        """
        output_name = job.outputs.benchmark_result.port_name
        download_outputs(
            job_name=job.name, output_name=output_name, download_path=output_dir
        )
        benchmark_result_path = os.path.join(
            output_dir,
            "named-outputs",
            "benchmark_result",
            "benchmark_result.json"
        )

        with open(benchmark_result_path, 'r') as fp:
            data = json.load(fp)
        assert isinstance(data['run_id'], str)
        assert len(data['run_id']) > 0

        if has_quality_step:
            assert len(data['quality_metrics']) > 0
            assert (
                'confusion_matrix' in data['quality_metrics'] or
                'bertscore' in data['quality_metrics']
            )
        else:
            assert len(data['quality_metrics']) == 0

        if has_perf_step:
            assert len(data['performance_metrics']) > 0
            assert 'latency_avg' in data['performance_metrics']
        else:
            assert len(data['performance_metrics']) == 0

        assert 'mlflow_parameters' in data
        assert len(data['mlflow_metrics']) > 0

        job_params_keys = [
            'inputs',
            'param',
            'run_id',
            'start_time',
            'end_time',
            'status',
            'maxRunDurationSeconds',
            'is_reused',
            'environment_version',
            'environment_name',
            'node_count',
            'vm_size'
        ]
        for key in job_params_keys:
            assert key in data['pipeline_params']['evaluation']

        assert len(data['simplified_pipeline_params']) >= 15
        assert 'evaluation.vm_size' in data['simplified_pipeline_params']
        assert data['model_name'] == 'finiteautomata-bertweet-base-sentiment-analysis'
        assert data['model_version'] == '8'
        assert data['model_registry'] == 'azureml'

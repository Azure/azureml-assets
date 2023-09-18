# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for Compute Performance Metrics Component."""

from typing import Optional
import json
import os
import uuid

from azure.ai.ml.entities import Job
from azure.ai.ml import Input

from utils import (
    load_yaml_pipeline,
    get_mlclient,
    Constants,
    download_outputs
)


class TestBatchOutputFormatterComponent:
    """Component test for batch output formatter."""

    EXP_NAME = "batch-inference-preparer-test"

    def test_batch_output_formatter(self, temp_dir: str):
        """Test method for batch inference preparer."""
        ml_client = get_mlclient()
        pipeline_job = self._get_pipeline_job(
            self.test_batch_output_formatter.__name__,
            "llama",
            'label',
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
        self._verify_output(
            pipeline_job, output_dir=out_dir
        )

    def _get_pipeline_job(
                self,
                display_name: str,
                model_type: str,
                label_key: str,
                temp_dir: Optional[str] = None,
            ) -> Job:
        pipeline_job = load_yaml_pipeline("batch_output_formatter.yaml")

        # avoid blob exists error when running pytest with multiple workers
        if temp_dir is not None:
            file_path = os.path.join(temp_dir, uuid.uuid4().hex + ".jsonl")
            with open(Constants.BATCH_INFERENCE_PREPARER_FILE_PATH, "r") as f:
                with open(file_path, "w") as f2:
                    f2.write(f.read())

        # set the pipeline inputs
        pipeline_job.inputs.input_dataset = Input(
            type="uri_file", path=file_path
        )
        pipeline_job.inputs.model_type = model_type
        pipeline_job.inputs.label_key = label_key

        pipeline_job.display_name = display_name

        return pipeline_job

    def _read_data(self, file_path):
        with open(file_path) as f:
            output_records = [json.loads(line) for line in f]
        return output_records

    def _check_columns_in_dfs(self, column_list, dfs):
        for df in dfs:
            for col in column_list:
                assert col in df

    def _check_output_data(self, output_dir, file_name, expected_col_list):
        output_file = os.path.join(output_dir, file_name)
        assert os.path.isfile(output_file)
        # Read the output file
        dfs = self._read_data(output_file)
        self._check_columns_in_dfs(expected_col_list, dfs)

    def _verify_output(self, job, output_dir):
        prediction_data = job.outputs.prediction_data.port_name
        perf_data = job.outputs.perf_data.port_name
        predict_ground_truth_data = job.outputs.predict_ground_truth_data.port_name
        for output_name in [prediction_data, perf_data, predict_ground_truth_data]:
            download_outputs(
                job_name=job.name, output_name=output_name, download_path=output_dir
            )
        self._check_output_data(output_dir, "prediction.jsonl", ["prediction"])
        self._check_output_data(output_dir, "predict_ground_truth_data.jsonl", ["ground_truth"])
        self._check_output_data(output_dir, "perf_data.jsonl", ["start", "end", "latency"])

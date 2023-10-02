# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for Batch inference preparer Component."""

from typing import Optional
import json
import os
import uuid

from azure.ai.ml.entities import Job
from azure.ai.ml import Input

from test_utils import (
    load_yaml_pipeline,
    get_mlclient,
    Constants,
    download_outputs,
    deploy_fake_test_endpoint_maybe
)


MLTABLE_CONTENTS = """type: mltable
paths:
  - pattern: ./*.json
transformations:
  - read_json_lines:
      encoding: utf8
      include_path_column: false
"""


class TestBatchBenchmarkScoreComponent:
    """Component test for batch score preparer."""

    EXP_NAME = "batch-benchmark-score-test"

    def test_batch_benchmark_score(self, temp_dir: str):
        """Test batch score preparer."""
        ml_client = get_mlclient()
        score_url, deployment_name = deploy_fake_test_endpoint_maybe(ml_client)
        pipeline_job = self._get_pipeline_job(
            self.test_batch_benchmark_score.__name__,
            score_url,
            '{"azureml-model-deployment": "' + deployment_name + '"}',
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
                online_endpoint_url: str,
                additional_headers: str,
                temp_dir: Optional[str] = None,
            ) -> Job:
        pipeline_job = load_yaml_pipeline("batch_benchmark_score.yaml")

        # avoid blob exists error when running pytest with multiple workers
        if temp_dir is not None:
            file_path = os.path.join(temp_dir, uuid.uuid4().hex + ".json")
            with open(Constants.BATCH_INFERENCE_FILE_PATH, "r") as f:
                with open(file_path, "w") as f2:
                    f2.write(f.read())
            with open(os.path.join(temp_dir, "MLTable"), "w") as f:
                f.writelines(MLTABLE_CONTENTS)

        # set the pipeline inputs
        pipeline_job.inputs.data_input_table = Input(
            type="mltable", path=temp_dir
        )
        pipeline_job.inputs.online_endpoint_url = online_endpoint_url
        pipeline_job.inputs.additional_headers = additional_headers

        pipeline_job.display_name = display_name
        pipeline_job.name = str(uuid.uuid4())

        return pipeline_job

    def _verify_output(self, job, output_dir):
        output_name = job.outputs.mini_batch_results_out_directory.port_name
        download_outputs(
            job_name=job.name, output_name=output_name, download_path=output_dir
        )
        output_dir = os.path.join(output_dir, "named-outputs", "mini_batch_results_out_directory")
        print(output_dir)
        for filename in os.listdir(output_dir):
            output_file_path = os.path.join(output_dir, filename)
            # Read the output file
            with open(output_file_path, "r") as f:
                output_records = [json.loads(line) for line in f]
            for r in output_records:
                assert r["status"] == 'success', "status should be success"

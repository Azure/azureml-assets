# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for Batch inference preparer Component."""

from typing import Optional
import json
import os
import uuid

from azure.ai.ml.entities import Job
from azure.ai.ml import Input

from utils import (
    get_mlclient,
    Constants,
    download_outputs,
    load_yaml_pipeline,
    deploy_fake_test_endpoint_maybe
)


class TestBatchBenchmarkInferenceComponent:
    """Component test for batch inference preparer."""

    EXP_NAME = "batch-benchmark-inference-test"

    def test_batch_benchmark_inference(self, temp_dir: str):
        """Test batch inference preparer."""
        ml_client = get_mlclient()
        score_url, deployment_name = deploy_fake_test_endpoint_maybe(ml_client)
        pipeline_job = self._get_pipeline_job(
            self.test_batch_benchmark_inference.__name__,
            score_url,
            '{"azureml-model-deployment": "' + deployment_name + '"}',
            '{'
            '   "input_data":'
            '   {'
            '       "input_string": ["###<prompt>"],'
            '       "parameters":'
            '       {'
            '           "temperature": 0.6,'
            '           "max_new_tokens": 100,'
            '           "do_sample": true'
            '       }'
            '   },'
            '   "_batch_request_metadata": ###<_batch_request_metadata>'
            '}',
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
                online_endpoint_url: str,
                additional_headers: str,
                batch_input_pattern: str,
                label_key: str,
                temp_dir: Optional[str] = None,
            ) -> Job:
        temp_yaml = self._create_inference_yaml()
        pipeline_job = load_yaml_pipeline('batch-benchmark-inference.yaml')
        os.remove(temp_yaml)
        # avoid blob exists error when running pytest with multiple workers
        if temp_dir is not None:
            file_path = os.path.join(temp_dir, uuid.uuid4().hex + ".jsonl")
            with open(Constants.BATCH_INFERENCE_PREPARER_FILE_PATH, "r") as f:
                with open(file_path, "w") as f2:
                    f2.write(f.read())

        # set the pipeline inputs
        pipeline_job.inputs.input_dataset = Input(
            type="uri_folder", path=temp_dir
        )
        pipeline_job.inputs.online_endpoint_url = online_endpoint_url
        pipeline_job.inputs.additional_headers = additional_headers
        pipeline_job.inputs.initial_worker_count = 1
        pipeline_job.inputs.max_worker_count = 10
        pipeline_job.inputs.batch_input_pattern = batch_input_pattern
        pipeline_job.inputs.label_key = label_key
        pipeline_job.name = str(uuid.uuid4())
        pipeline_job.display_name = display_name

        return pipeline_job

    def _create_inference_yaml(self):
        original_yml_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "components", "batch-benchmark-inference", "spec.yaml")
        new_yaml_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "pipelines", "batch-benchmark-inference.yaml")
        new_lines = []
        current_section = "main"
        with open(original_yml_path) as f1:
            for line1 in f1.readlines():
                current_section = self._update_currect_section(current_section, line1)
                if not self._should_keep_line(current_section, line1):
                    continue
                new_lines.extend(self._get_updated_lines(current_section, line1))
        # add compute
        new_lines.append('settings:\n')
        new_lines.append('  default_compute: azureml:serverless\n')
        with open(new_yaml_path, "w") as f2:
            f2.writelines(new_lines)
        return new_yaml_path

    def _get_updated_lines(self, current_section, line):
        param_mapping_dict = {
            "input_dataset:": ['  input_dataset:\n', '    type: uri_folder\n', '    path: ../data/\n'],
            "batch_input_pattern:": ["  batch_input_pattern: wrong_pattern\n"],
            "online_endpoint_url:": ["  online_endpoint_url: a_bad_url\n"],
            "debug_mode:": ["  debug_mode: false\n"],
            "additional_headers:": ["  additional_headers: 'some_header'\n"],
            "ensure_ascii:": ["  ensure_ascii: false\n"],
            "max_retry_time_interval:": ["  max_retry_time_interval: 600\n"],
            "initial_worker_count:": ["  initial_worker_count: 1\n"],
            "max_worker_count:": ["  max_worker_count: 10\n"],
            "instance_count:": ["  instance_count: 1\n"],
            "max_concurrency_per_instance:": ["  max_concurrency_per_instance: 1\n"],
            "ground_truth_input:": ['  ground_truth_input:\n', '    type: uri_folder\n', '    path: ../data/\n'],
            "metadata_key:": ["  metadata_key: _batch_request_metadata\n"],
            "label_key:": ["  label_key: label\n"],
            "n_samples:": ["  n_samples: 10\n"],
            "prediction_data:": [
                '  prediction_data:\n', '    type: uri_file\n',
                '    path: azureml://datastores/${{default_datastore}}/paths/${{name}}/prediction.jsonl\n'],
            "perf_data:": [
                '  perf_data:\n', '    type: uri_file\n',
                '    path: azureml://datastores/${{default_datastore}}/paths/${{name}}/perf_data.jsonl\n'],
            "ground_truth_data:": [
                '  ground_truth_data:\n', '    type: uri_file\n',
                '    path: azureml://datastores/${{default_datastore}}'
                '/paths/${{name}}/ground_truth_data.jsonl\n']
        }
        if current_section == "inputs" or current_section == "outputs":
            line_key = self._get_yml_key(line)
            if line_key in param_mapping_dict:
                return param_mapping_dict[line_key]
        if current_section == "jobs":
            for component in ["batch_inference_preparer", "batch_benchmark_score", "batch_output_formatter"]:
                if f"azureml:{component}" in line:
                    return [f"    component: ../../components/{component.replace('_', '-')}/spec.yaml\n"]
        return [line]

    def _get_yml_key(self, line):
        if line.strip():
            return line.strip().split()[0]
        return line

    def _update_currect_section(self, current_section, line):
        if line.startswith("inputs:"):
            return "inputs"
        if line.startswith("outputs:"):
            return "outputs"
        if line.startswith("jobs:"):
            return "jobs"
        return current_section

    def _should_keep_line(self, current_section, line):
        if current_section == "main":
            return self._get_yml_key(line) not in {"version:", "name:"}
        if current_section == "inputs" or current_section == "outputs":
            if line.startswith("      "):
                return False
            return self._get_yml_key(line) not in {'optional:', 'type:', 'default:', 'description:'}
        return True

    def _read_data(self, file_path):
        with open(file_path) as f:
            output_records = [json.loads(line) for line in f]
        return output_records

    def _check_columns_in_dfs(self, column_list, dfs):
        for df in dfs:
            for col in column_list:
                assert col in df, f"{col} not found in df."

    def _check_output_data(self, output_dir, file_name, expected_col_list):
        output_file = os.path.join(output_dir, file_name)
        assert os.path.isfile(output_file), f"{output_file} is not a file"
        # Read the output file
        dfs = self._read_data(output_file)
        self._check_columns_in_dfs(expected_col_list, dfs)

    def _verify_output(self, job, output_dir):
        prediction_data = job.outputs.prediction_data.port_name
        perf_data = job.outputs.perf_data.port_name
        ground_truth_data = job.outputs.ground_truth_data.port_name
        for output_name in [prediction_data, perf_data, ground_truth_data]:
            download_outputs(
                job_name=job.name, output_name=output_name, download_path=output_dir
            )
        output_dir = os.path.join(output_dir, "named-outputs")
        self._check_output_data(
            os.path.join(output_dir, "prediction_data"), "prediction.jsonl", ["prediction"])
        self._check_output_data(
            os.path.join(output_dir, "perf_data"), "perf_data.jsonl", ["start", "end", "latency"])
        self._check_output_data(
            os.path.join(
                output_dir, "ground_truth_data"), "ground_truth_data.jsonl", ["ground_truth"])
